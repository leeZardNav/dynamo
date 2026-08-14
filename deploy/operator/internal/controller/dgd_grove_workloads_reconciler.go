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
	"fmt"
	"strconv"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// groveWorkloadsReconciler owns the complete provider workload sequence while
// exposing no dependency on the top-level DGD controller.
type groveWorkloadsReconciler struct {
	syncer          dgdResourceSyncer
	reader          client.Reader
	renderer        *groveWorkloadRenderer
	scaler          *groveScaler
	stableResources *groveStableResourcesReconciler
}

const groveProviderOverrideFieldOwner = "dynamo-operator-grove-provider-override"

func newGroveWorkloadsReconciler(
	kubeClient client.Client,
	recorder events.EventRecorder,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	dockerSecretRetriever DockerSecretRetriever,
	scaleClient scale.ScalesGetter,
) *groveWorkloadsReconciler {
	return &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, recorder),
		reader: kubeClient,
		renderer: newGroveWorkloadRenderer(
			kubeClient,
			config,
			runtimeConfig,
			dockerSecretRetriever,
		),
		scaler:          newGroveScaler(scaleClient),
		stableResources: newGroveStableResourcesReconciler(kubeClient, recorder, config),
	}
}

func (r *groveWorkloadsReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) (ReconcileResult, error) {
	logger := log.FromContext(ctx)

	desiredPodCliqueSet, err := r.renderer.Render(
		ctx,
		dgd,
		restartState,
		checkpointInfos,
	)
	if err != nil {
		logger.Error(err, "failed to generate the Grove GangSet")
		return ReconcileResult{}, fmt.Errorf("failed to generate the Grove GangSet: %w", err)
	}
	renderDeployment := groveRenderDeployment(dgd, desiredPodCliqueSet)

	// Keep the existing typed path unless sparse provider-native fields require SSA.
	var syncedPodCliqueSet *grovev1alpha1.PodCliqueSet
	if provideroverride.HasGroveOverrides(dgd) {
		providerObject, applyErr := provideroverride.ApplyGroveOverrides(dgd, desiredPodCliqueSet)
		if applyErr != nil {
			return ReconcileResult{}, fmt.Errorf("failed to apply Grove provider overrides: %w", applyErr)
		}
		syncedPodCliqueSet, err = r.reconcileProviderOverridePodCliqueSet(ctx, dgd, providerObject)
	} else {
		syncedPodCliqueSet, err = r.reconcilePodCliqueSet(ctx, dgd, desiredPodCliqueSet)
	}
	if err != nil {
		logger.Error(err, "failed to reconcile the Grove PodCliqueSet")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile the Grove PodCliqueSet: %w", err)
	}

	if err := r.scaler.Reconcile(ctx, dgd, checkpointInfos); err != nil {
		logger.Error(err, "failed to reconcile Grove scaling")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile Grove scaling: %w", err)
	}

	stableResources, err := r.stableResources.Reconcile(ctx, dgd, renderDeployment)
	if err != nil {
		return ReconcileResult{}, err
	}

	podCliqueSetResource, readiness, err := r.observePodCliqueSetReadiness(
		ctx,
		dgd,
		syncedPodCliqueSet,
	)
	if err != nil {
		return ReconcileResult{}, err
	}

	resources := append(stableResources, podCliqueSetResource)
	return checkGroveResourcesReadiness(resources, readiness.Classification), nil
}

// reconcileProviderOverridePodCliqueSet validates the complete rendered
// provider object through the API server before mutating the live object. The
// same unstructured payload is then applied atomically with one field manager,
// preserving provider fields unknown to Dynamo's compiled Grove Go types. dgd
// and desired must not be nil.
func (r *groveWorkloadsReconciler) reconcileProviderOverridePodCliqueSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *unstructured.Unstructured,
) (*grovev1alpha1.PodCliqueSet, error) {
	// Attach ownership before hashing so the dry-run and apply payloads match.
	if err := ctrl.SetControllerReference(dgd, desired, r.syncer.Scheme()); err != nil {
		return nil, fmt.Errorf("set PodCliqueSet controller reference: %w", err)
	}

	// Record the desired program hash on the object managed through SSA.
	desiredHash, err := commoncontroller.GetSpecHash(desired)
	if err != nil {
		return nil, fmt.Errorf("hash PodCliqueSet provider program: %w", err)
	}
	desiredAnnotations := desired.GetAnnotations()
	if desiredAnnotations == nil {
		desiredAnnotations = make(map[string]string)
	}
	desiredAnnotations[commoncontroller.NvidiaAnnotationHashKey] = desiredHash
	desired.SetAnnotations(desiredAnnotations)

	// Reuse an unchanged live object without issuing another API read.
	key := types.NamespacedName{Name: desired.GetName(), Namespace: desired.GetNamespace()}
	live := &unstructured.Unstructured{}
	live.SetGroupVersionKind(desired.GroupVersionKind())
	if getErr := r.syncer.Get(ctx, key, live); getErr == nil {
		annotations := live.GetAnnotations()
		if annotations[commoncontroller.NvidiaAnnotationHashKey] == desiredHash &&
			annotations[commoncontroller.NvidiaAnnotationGenerationKey] == strconv.FormatInt(live.GetGeneration(), 10) {
			synced := &grovev1alpha1.PodCliqueSet{}
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(live.Object, synced); err != nil {
				return nil, fmt.Errorf("convert unchanged PodCliqueSet %s: %w", key, err)
			}
			return synced, nil
		}
	} else if !apierrors.IsNotFound(getErr) {
		return nil, fmt.Errorf("get PodCliqueSet provider program %s: %w", key, getErr)
	}

	// Ask the installed CRD and webhooks to validate the exact future-aware payload.
	patchOptions := []client.PatchOption{
		client.FieldOwner(groveProviderOverrideFieldOwner),
		client.ForceOwnership,
		client.FieldValidation(metav1.FieldValidationStrict),
	}
	if err := r.syncer.Patch(
		ctx,
		desired.DeepCopy(),
		client.Apply,
		append(patchOptions, client.DryRunAll)...,
	); err != nil {
		return nil, fmt.Errorf("dry-run PodCliqueSet provider program: %w", err)
	}

	// Apply the exact payload only after server-side validation succeeds.
	if err := r.syncer.Patch(ctx, desired, client.Apply, patchOptions...); err != nil {
		return nil, fmt.Errorf("apply PodCliqueSet provider program: %w", err)
	}

	// Checkpoint the server-returned generation without a read-after-write.
	original := desired.DeepCopy()
	annotations := desired.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[commoncontroller.NvidiaAnnotationHashKey] = desiredHash
	annotations[commoncontroller.NvidiaAnnotationGenerationKey] = strconv.FormatInt(desired.GetGeneration(), 10)
	desired.SetAnnotations(annotations)
	if err := r.syncer.Patch(ctx, desired, client.MergeFrom(original)); err != nil {
		return nil, fmt.Errorf("checkpoint applied PodCliqueSet provider program: %w", err)
	}

	// Return the typed view used by the existing Grove readiness path.
	synced := &grovev1alpha1.PodCliqueSet{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(desired.Object, synced); err != nil {
		return nil, fmt.Errorf("convert applied PodCliqueSet %s: %w", key, err)
	}
	return synced, nil
}

func (r *groveWorkloadsReconciler) reconcilePodCliqueSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *grovev1alpha1.PodCliqueSet,
) (*grovev1alpha1.PodCliqueSet, error) {
	_, synced, err := commoncontroller.SyncResource(
		ctx,
		&r.syncer,
		dgd,
		func(context.Context) (*grovev1alpha1.PodCliqueSet, bool, error) {
			return desired, false, nil
		},
	)
	if err != nil {
		return nil, err
	}
	return synced, nil
}

// observePodCliqueSetReadiness takes the authoritative Grove snapshot after
// structural and scale reconciliation, then adapts it to the common Resource
// readiness interface without further Kubernetes reads.
func (r *groveWorkloadsReconciler) observePodCliqueSetReadiness(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	podCliqueSet *grovev1alpha1.PodCliqueSet,
) (*commoncontroller.Resource, dynamo.GroveReadiness, error) {
	readiness, err := dynamo.EvaluateGroveReadiness(ctx, r.reader, dgd)
	if err != nil {
		return nil, dynamo.GroveReadiness{}, err
	}
	resource, err := commoncontroller.NewResourceWithComponentStatuses(
		podCliqueSet,
		func() (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
			return readiness.Ready, readiness.Message, readiness.ComponentStatuses
		},
	)
	if err != nil {
		return nil, dynamo.GroveReadiness{}, fmt.Errorf("failed to create the Grove PodCliqueSet resource: %w", err)
	}
	return resource, readiness, nil
}

func checkGroveResourcesReadiness(
	resources []Resource,
	classification string,
) ReconcileResult {
	result := checkResourcesReadiness(resources)
	if result.State == nvidiacomv1beta1.DGDStateSuccessful {
		return result
	}
	if classification != "" {
		result.Reason = Reason(classification)
	}
	return result
}
