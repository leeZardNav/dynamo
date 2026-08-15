//go:build !clustertest

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
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	grovecrds "github.com/ai-dynamo/grove/operator/api/core/v1alpha1/crds"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"
)

func TestGroveProviderOverrideDryRunUsesInstalledCRDSchema(t *testing.T) {
	ctx := context.Background()

	t.Log("Install the embedded Grove PodCliqueSet CRD in the shared API server")
	env := sharedEnv.ForTest(t)
	crd := &apiextensionsv1.CustomResourceDefinition{}
	require.NoError(t, yaml.Unmarshal([]byte(grovecrds.PodCliqueSetCRD()), crd))
	crdClient := env.Client()
	if err := crdClient.Create(ctx, crd); err != nil {
		require.True(t, apierrors.IsAlreadyExists(err), "create PodCliqueSet CRD: %v", err)
	} else {
		t.Cleanup(func() {
			if err := crdClient.Delete(context.Background(), crd); err != nil && !apierrors.IsNotFound(err) {
				t.Errorf("delete PodCliqueSet CRD: %v", err)
			}
		})
	}
	require.Eventually(t, func() bool {
		current := &apiextensionsv1.CustomResourceDefinition{}
		if err := crdClient.Get(ctx, client.ObjectKey{Name: crd.Name}, current); err != nil {
			return false
		}
		for _, condition := range current.Status.Conditions {
			if condition.Type == apiextensionsv1.Established && condition.Status == apiextensionsv1.ConditionTrue {
				return true
			}
		}
		return false
	}, 10*time.Second, 100*time.Millisecond)

	t.Log("Create one valid live PodCliqueSet through a client that discovered the new CRD")
	providerClient, err := client.New(env.RESTConfig(), client.Options{Scheme: crdClient.Scheme()})
	require.NoError(t, err)
	live := &grovev1alpha1.PodCliqueSet{
		TypeMeta: metav1.TypeMeta{APIVersion: provideroverride.GroveAPIVersion, Kind: provideroverride.TargetPodCliqueSet},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "provider-dry-run",
			Namespace: env.Namespace(),
		},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Replicas: 1,
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{
					Name: "frontend",
					Spec: grovev1alpha1.PodCliqueSpec{
						RoleName:     "frontend",
						Replicas:     1,
						MinAvailable: ptr.To(int32(1)),
						PodSpec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "main", Image: "registry.example/frontend:test"}},
						},
					},
				}},
			},
		},
	}
	require.NoError(t, providerClient.Create(ctx, live))

	t.Log("Render a sparse field that Dynamo preserves but this installed Grove CRD does not know")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "provider-dry-run",
			Namespace: env.Namespace(),
			UID:       types.UID("provider-dry-run-uid"),
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			ProviderOverride: &nvidiacomv1beta1.ProviderOverride{
				APIVersion: provideroverride.GroveAPIVersion,
				Target:     provideroverride.TargetPodCliqueSet,
				Value: apiextensionsv1.JSON{Raw: []byte(
					`{"spec":{"template":{"topologyConstraint":{"futureProviderField":"unsupported"}}}}`,
				)},
			},
		},
	}
	rendered := live.DeepCopy()
	rendered.ResourceVersion = ""
	rendered.UID = ""
	rendered.CreationTimestamp = metav1.Time{}
	rendered.Generation = 0
	rendered.ManagedFields = nil
	desired, err := provideroverride.ApplyGroveOverrides(dgd, rendered)
	require.NoError(t, err)
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(providerClient, events.NewFakeRecorder(10)),
	}

	t.Log("Reject the complete provider program during server-side dry-run")
	_, err = reconciler.reconcileProviderOverridePodCliqueSet(ctx, dgd, desired)
	require.ErrorContains(t, err, "dry-run PodCliqueSet provider program")
	require.ErrorContains(t, err, "futureProviderField")

	t.Log("Verify the failed dry-run did not mutate the live workload")
	unchanged := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, providerClient.Get(ctx, client.ObjectKeyFromObject(live), unchanged))
	assert.Equal(t, live.ResourceVersion, unchanged.ResourceVersion)
	assert.Equal(t, live.Generation, unchanged.Generation)
	assert.Nil(t, unchanged.Spec.Template.TopologyConstraint)

	t.Log("Apply a supported provider fragment through the same API-server path")
	dgd.Name = "provider-dry-run-valid"
	dgd.UID = types.UID("provider-dry-run-valid-uid")
	dgd.Spec.ProviderOverride.Value.Raw = []byte(
		`{"spec":{"template":{"topologyConstraint":{"pack":{"required":"rack"}}}}}`,
	)
	rendered.Name = dgd.Name
	supported, err := provideroverride.ApplyGroveOverrides(dgd, rendered)
	require.NoError(t, err)
	synced, err := reconciler.reconcileProviderOverridePodCliqueSet(ctx, dgd, supported)
	require.NoError(t, err)
	require.NotNil(t, synced.Spec.Template.TopologyConstraint)
	require.NotNil(t, synced.Spec.Template.TopologyConstraint.Pack)
	assert.Equal(t, grovev1alpha1.TopologyDomain("rack"), synced.Spec.Template.TopologyConstraint.Pack.RequiredDomain)
}
