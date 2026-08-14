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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGroveWorkloadsReconciler_EvaluatesReadinessOnce(t *testing.T) {
	t.Log("Build a Grove DGD that uses the provider-override reconciliation path")
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
	dgd.Spec.ProviderOverride = &nvidiacomv1beta1.ProviderOverride{
		APIVersion: provideroverride.GroveAPIVersion,
		Target:     provideroverride.TargetPodCliqueSet,
		Value: apiextensionsv1.JSON{Raw: []byte(
			`{"spec":{"template":{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"rack"}}}}}`,
		)},
	}

	t.Log("Expose one ready PodClique and record readiness reads after scaling")
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

	t.Log("Wire the Grove program with a recording scale client")
	podCliqueReads := 0
	scaleClient := &recordingGroveScaleClient{}
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, podClique).
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
				return reader.Get(ctx, key, object, options...)
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

	t.Log("Verify readiness was evaluated exactly once after scaling")
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.State)
	assert.Equal(t, 1, podCliqueReads)
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

	t.Log("Record dry-run, apply, and checkpoint patches")
	var patchDryRuns []bool
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
				if _, ok := obj.(*unstructured.Unstructured); ok {
					providerProgramReads++
				}
				return reader.Get(ctx, key, obj, opts...)
			},
			Patch: func(
				ctx context.Context,
				writer client.WithWatch,
				obj client.Object,
				patch client.Patch,
				opts ...client.PatchOption,
			) error {
				options := (&client.PatchOptions{}).ApplyOptions(opts)
				patchDryRuns = append(patchDryRuns, len(options.DryRun) != 0)
				if payload, ok := obj.(*unstructured.Unstructured); ok {
					cliques, found, nestedErr := unstructured.NestedSlice(payload.Object, "spec", "template", "cliques")
					if nestedErr == nil && found && len(cliques) != 0 {
						clique := cliques[0].(map[string]interface{})
						providerFieldSent = providerFieldSent || clique["topologyConstraint"].(map[string]interface{})["futureProviderField"] == "preserved"
					}
				}
				return writer.Patch(ctx, obj, patch, opts...)
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}

	t.Log("Reconcile the provider program through strict dry-run and SSA")
	synced, err := reconciler.reconcileProviderOverridePodCliqueSet(context.Background(), dgd, desired)

	t.Log("Verify validation preceded mutation without reading the successful write back")
	require.NoError(t, err)
	assert.Equal(t, []bool{true, false, false}, patchDryRuns)
	assert.Equal(t, 1, providerProgramReads)
	assert.Equal(t, int32(1), synced.Spec.Replicas)

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
	actualPatches := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd, existing).
		WithInterceptorFuncs(interceptor.Funcs{
			Patch: func(
				ctx context.Context,
				writer client.WithWatch,
				obj client.Object,
				patch client.Patch,
				opts ...client.PatchOption,
			) error {
				options := (&client.PatchOptions{}).ApplyOptions(opts)
				if len(options.DryRun) != 0 {
					return errors.New("provider webhook unavailable")
				}
				actualPatches++
				return writer.Patch(ctx, obj, patch, opts...)
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}

	t.Log("Attempt to reconcile the invalid provider program")
	_, err := reconciler.reconcileProviderOverridePodCliqueSet(context.Background(), dgd, desired)

	t.Log("Verify reconciliation stopped before a mutating patch")
	require.ErrorContains(t, err, "dry-run PodCliqueSet provider program")
	assert.Zero(t, actualPatches)

	t.Log("Verify the previously applied workload was preserved")
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, int32(1), live.Spec.Replicas)
}
