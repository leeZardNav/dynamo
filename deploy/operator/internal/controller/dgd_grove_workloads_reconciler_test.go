/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"errors"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGroveWorkloadsReconciler_EvaluatesReadinessOnce(t *testing.T) {
	tests := []struct {
		name                   string
		currentOverride        bool
		previouslyManagedBySSA bool
		wantApplyCalls         int
	}{
		{name: "typed pathway", wantApplyCalls: 0},
		{name: "current provider override", currentOverride: true, wantApplyCalls: 2},
		{name: "removed provider override", previouslyManagedBySSA: true, wantApplyCalls: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a Grove DGD for the selected reconciliation pathway")
			dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"frontend": {
							ComponentType: consts.ComponentTypeFrontend,
							Replicas:      ptr.To(int32(1)),
						},
					},
				},
			})
			if tt.currentOverride {
				dgd.Spec.ProviderOverride = &nvidiacomv1beta1.ProviderOverride{
					APIVersion: provideroverride.GroveAPIVersion,
					Target:     provideroverride.TargetPodCliqueSet,
					Value: apiextensionsv1.JSON{Raw: []byte(
						`{"spec":{"template":{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"rack"}}}}}`,
					)},
				}
			}

			t.Log("Expose one ready PodClique and any durable SSA pathway marker")
			podClique := &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "graph-0-frontend",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1},
				Status: grovev1alpha1.PodCliqueStatus{
					Replicas:           1,
					ReadyReplicas:      1,
					UpdatedReplicas:    1,
					ScheduledReplicas:  1,
					ObservedGeneration: ptr.To(int64(1)),
				},
			}
			objects := []client.Object{dgd, podClique}
			if tt.previouslyManagedBySSA {
				objects = append(objects, &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{
					Name:      "graph",
					Namespace: "default",
				}})
			}

			t.Log("Wire the Grove program with recording scale and apply clients")
			podCliqueReads := 0
			applyCalls := 0
			scaleClient := &recordingGroveScaleClient{}
			kubeClient := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(objects...).
				WithStatusSubresource(dgd, podClique).
				WithInterceptorFuncs(interceptor.Funcs{
					Get: func(
						ctx context.Context,
						reader client.WithWatch,
						key client.ObjectKey,
						object client.Object,
						options ...client.GetOption,
					) error {
						if _, ok := object.(*grovev1alpha1.PodClique); ok {
							require.Len(t, scaleClient.updates, 1, "readiness must be observed after scaling")
							podCliqueReads++
						}
						getErr := reader.Get(ctx, key, object, options...)
						if getErr == nil && tt.previouslyManagedBySSA {
							if podCliqueSet, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
								podCliqueSet.ManagedFields = []metav1.ManagedFieldsEntry{{
									Manager: groveProviderOverrideFieldOwner,
								}}
							}
						}
						return getErr
					},
					Apply: func(
						ctx context.Context,
						writer client.WithWatch,
						object runtime.ApplyConfiguration,
						options ...client.ApplyOption,
					) error {
						applyCalls++
						return writer.Apply(ctx, object, options...)
					},
				}).
				Build()
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:        kubeClient,
				Config:        &configv1alpha1.OperatorConfiguration{},
				Recorder:      events.NewFakeRecorder(10),
				RuntimeConfig: &commoncontroller.RuntimeConfig{},
				ScaleClient:   scaleClient,
				DockerSecretRetriever: &mockDockerSecretRetriever{
					GetSecretsFunc: func(string, string) ([]string, error) {
						return nil, nil
					},
				},
			}

			t.Log("Reconcile the complete Grove workload program")
			result, err := reconciler.newGroveProgram().workloads.Reconcile(
				context.Background(),
				dgd,
				nil,
				nil,
			)

			t.Log("Verify pathway selection and one readiness observation after scaling")
			require.NoError(t, err)
			assert.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.State)
			assert.Equal(t, tt.wantApplyCalls, applyCalls)
			assert.Equal(t, 1, podCliqueReads)
		})
	}
}

func TestGroveProviderOverrideReconcileDryRunsBeforeApply(t *testing.T) {
	t.Log("Build a future-aware unstructured Grove provider program")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1beta1.GroupVersion.String(),
			Kind:       "DynamoGraphDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			UID:       types.UID("graph-uid"),
		},
	}
	desired := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "grove.io/v1alpha1",
		"kind":       "PodCliqueSet",
		"metadata": map[string]interface{}{
			"name":      "graph",
			"namespace": "default",
		},
		"spec": map[string]interface{}{
			"replicas": int64(1),
			"template": map[string]interface{}{
				"cliques": []interface{}{
					map[string]interface{}{
						"name": "frontend",
						"spec": map[string]interface{}{
							"roleName":     "frontend",
							"replicas":     int64(1),
							"minAvailable": int64(1),
							"podSpec": map[string]interface{}{
								"containers": []interface{}{},
							},
						},
						"topologyConstraint": map[string]interface{}{
							"futureProviderField": "preserved",
						},
					},
				},
			},
		},
	}}

	t.Log("Record dry-run, apply, and checkpoint operations")
	var applyDryRuns []bool
	providerProgramReads := 0
	providerFieldSent := false
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(
				ctx context.Context,
				reader client.WithWatch,
				key client.ObjectKey,
				obj client.Object,
				opts ...client.GetOption,
			) error {
				if _, ok := obj.(*grovev1alpha1.PodCliqueSet); ok {
					providerProgramReads++
				}
				return reader.Get(ctx, key, obj, opts...)
			},
			Apply: func(
				ctx context.Context,
				writer client.WithWatch,
				obj runtime.ApplyConfiguration,
				opts ...client.ApplyOption,
			) error {
				options := (&client.ApplyOptions{}).ApplyOptions(opts)
				applyDryRuns = append(applyDryRuns, len(options.DryRun) != 0)

				// Inspect the future-aware payload without relying on unchecked shapes.
				payload, ok := obj.(interface{ UnstructuredContent() map[string]interface{} })
				require.True(t, ok, "apply configuration must expose unstructured content")
				cliques, found, nestedErr := unstructured.NestedSlice(payload.UnstructuredContent(), "spec", "template", "cliques")
				require.NoError(t, nestedErr)
				require.True(t, found)
				require.NotEmpty(t, cliques)
				clique, ok := cliques[0].(map[string]interface{})
				require.True(t, ok, "first clique must be an object")
				constraint, ok := clique["topologyConstraint"].(map[string]interface{})
				require.True(t, ok, "topologyConstraint must be an object")
				providerFieldSent = providerFieldSent || constraint["futureProviderField"] == "preserved"
				return writer.Apply(ctx, obj, opts...)
			},
		}).
		Build()
	recorder := events.NewFakeRecorder(10)
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, recorder),
	}

	t.Log("Reconcile the provider program through strict dry-run and SSA")
	synced, err := reconciler.reconcileProviderOverridePodCliqueSet(context.Background(), dgd, desired)

	t.Log("Verify validation preceded mutation without reading the successful write back")
	require.NoError(t, err)
	assert.Equal(t, []bool{true, false}, applyDryRuns)
	assert.Equal(t, 1, providerProgramReads)
	assert.Equal(t, int32(1), synced.Spec.Replicas)
	select {
	case event := <-recorder.Events:
		assert.Contains(t, event, "CreatePodCliqueSet")
	default:
		t.Fatal("provider workload mutation event was not emitted")
	}

	t.Log("Verify the SSA payload retained the unknown provider field")
	assert.True(t, providerFieldSent, "unstructured provider field must be present in the SSA payload")

	t.Log("Verify the applied workload remains readable through the typed Grove API")
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, int32(1), live.Spec.Replicas)
}

func TestGroveProviderOverrideReconcileDoesNotMutateWhenDryRunFails(t *testing.T) {
	t.Log("Build an existing workload and a materially different desired program")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			UID:       types.UID("graph-uid"),
		},
	}
	existing := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec:       grovev1alpha1.PodCliqueSetSpec{Replicas: 1},
	}
	desired := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "grove.io/v1alpha1",
		"kind":       "PodCliqueSet",
		"metadata": map[string]interface{}{
			"name":      "graph",
			"namespace": "default",
		},
		"spec": map[string]interface{}{
			"replicas": int64(7),
			"template": map[string]interface{}{"cliques": []interface{}{}},
		},
	}}
	t.Log("Fail the API-server dry-run and record any real mutation attempt")
	actualApplies := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd, existing).
		WithInterceptorFuncs(interceptor.Funcs{
			Apply: func(
				ctx context.Context,
				writer client.WithWatch,
				obj runtime.ApplyConfiguration,
				opts ...client.ApplyOption,
			) error {
				options := (&client.ApplyOptions{}).ApplyOptions(opts)
				if len(options.DryRun) != 0 {
					return errors.New("provider webhook unavailable")
				}
				actualApplies++
				return writer.Apply(ctx, obj, opts...)
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}

	t.Log("Attempt to reconcile the invalid provider program")
	_, err := reconciler.reconcileProviderOverridePodCliqueSet(context.Background(), dgd, desired)

	t.Log("Verify reconciliation stopped before a mutating apply")
	require.ErrorContains(t, err, "dry-run PodCliqueSet provider program")
	assert.Zero(t, actualApplies)

	t.Log("Verify the previously applied workload was preserved")
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, int32(1), live.Spec.Replicas)
}
