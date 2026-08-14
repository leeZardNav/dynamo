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

package provideroverride

import (
	"encoding/json"
	"fmt"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	jsonpatch "github.com/evanphx/json-patch/v5"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	sigsjson "sigs.k8s.io/json"
)

// HasGroveOverrides reports whether a DGD contains a provider-native fragment
// at any Grove provider context. dgd must not be nil.
func HasGroveOverrides(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	// Check the root before traversing component and role contexts.
	if dgd.Spec.ProviderOverride != nil {
		return true
	}

	// Stop as soon as any component-level provider context is populated.
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.ProviderOverride != nil {
			return true
		}
		if component.Multinode == nil {
			continue
		}
		if component.Multinode.Leader != nil && component.Multinode.Leader.ProviderOverride != nil {
			return true
		}
		if component.Multinode.Worker != nil && component.Multinode.Worker.ProviderOverride != nil {
			return true
		}
	}
	return false
}

// ApplyGroveOverrides converts a fully rendered PodCliqueSet to unstructured
// form and overlays each sparse provider-native fragment at its resolved
// destination. Keeping the result unstructured preserves provider fields that
// are newer than the Grove Go types compiled into Dynamo. dgd and desired must
// not be nil.
func ApplyGroveOverrides(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *grovev1alpha1.PodCliqueSet,
) (*unstructured.Unstructured, error) {
	// Preserve the rendered object as unstructured data before applying fragments.
	object, err := runtime.DefaultUnstructuredConverter.ToUnstructured(desired)
	if err != nil {
		return nil, fmt.Errorf("convert rendered PodCliqueSet to unstructured: %w", err)
	}
	result := &unstructured.Unstructured{Object: object}
	result.SetAPIVersion(GroveAPIVersion)
	result.SetKind(TargetPodCliqueSet)

	// Apply the root fragment before the more specific component destinations.
	if err := applyGroveRootOverride(result, dgd.Spec.ProviderOverride); err != nil {
		return nil, fmt.Errorf("spec.providerOverride: %w", err)
	}

	// Resolve each component and role fragment to its generated Grove destination.
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentPath := fmt.Sprintf("spec.components[%d]", i)
		if err := applyGroveComponentOverride(result, component, component.ProviderOverride); err != nil {
			return nil, fmt.Errorf("%s.providerOverride: %w", componentPath, err)
		}
		if component.Multinode == nil {
			continue
		}
		if component.Multinode.Leader != nil {
			if err := applyGroveRoleOverride(
				result,
				component,
				ScopeMultinodeLeader,
				component.Multinode.Leader.ProviderOverride,
			); err != nil {
				return nil, fmt.Errorf("%s.multinode.leader.providerOverride: %w", componentPath, err)
			}
		}
		if component.Multinode.Worker != nil {
			if err := applyGroveRoleOverride(
				result,
				component,
				ScopeMultinodeWorker,
				component.Multinode.Worker.ProviderOverride,
			); err != nil {
				return nil, fmt.Errorf("%s.multinode.worker.providerOverride: %w", componentPath, err)
			}
		}
	}
	return result, nil
}

// applyGroveRootOverride applies an optional root fragment. result must not be
// nil; a nil override is an intentional no-op.
func applyGroveRootOverride(result *unstructured.Unstructured, override *nvidiacomv1beta1.ProviderOverride) error {
	// A missing fragment is an intentional no-op for this provider context.
	if override == nil {
		return nil
	}

	// Verify the persisted identity before merging the root object fragment.
	if err := validateOverrideIdentity(override, ScopeRoot, nil); err != nil {
		return err
	}

	// Overlay the sparse root fragment with JSON Merge Patch semantics.
	patched, err := mergeJSONObjects(result.Object, override.Value.Raw)
	if err != nil {
		return fmt.Errorf("merge value: %w", err)
	}
	result.Object = patched
	return nil
}

// applyGroveComponentOverride applies an optional component fragment. result
// and component must not be nil; a nil override is an intentional no-op.
func applyGroveComponentOverride(
	result *unstructured.Unstructured,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	override *nvidiacomv1beta1.ProviderOverride,
) error {
	// A missing fragment is an intentional no-op for this provider context.
	if override == nil {
		return nil
	}

	// Verify the persisted identity before selecting its generated list.
	if err := validateOverrideIdentity(override, ScopeComponent, component); err != nil {
		return err
	}

	// Route the fragment to the list implied by its embedded Grove target.
	name := strings.ToLower(component.ComponentName)
	switch override.Target {
	case TargetPodCliqueTemplateSpec:
		return patchNamedGroveTemplate(
			result,
			[]string{"spec", "template", "cliques"},
			name,
			override.Value.Raw,
		)
	case TargetPodCliqueScalingGroupConfig:
		return patchNamedGroveTemplate(
			result,
			[]string{"spec", "template", "podCliqueScalingGroups"},
			name,
			override.Value.Raw,
		)
	default:
		return fmt.Errorf("unsupported target %q", override.Target)
	}
}

// applyGroveRoleOverride applies an optional multinode role fragment. result
// and component must not be nil; a nil override is an intentional no-op.
func applyGroveRoleOverride(
	result *unstructured.Unstructured,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	scope Scope,
	override *nvidiacomv1beta1.ProviderOverride,
) error {
	// A missing fragment is an intentional no-op for this provider context.
	if override == nil {
		return nil
	}

	// Verify the persisted role identity before resolving its generated name.
	if err := validateOverrideIdentity(override, scope, component); err != nil {
		return err
	}
	suffix := consts.GroveRoleSuffixLeader
	if scope == ScopeMultinodeWorker {
		suffix = consts.GroveRoleSuffixWorker
	}

	// Patch the PCLQ template named for the selected multinode role.
	return patchNamedGroveTemplate(
		result,
		[]string{"spec", "template", "cliques"},
		strings.ToLower(component.ComponentName+"-"+suffix),
		override.Value.Raw,
	)
}

// validateOverrideIdentity checks one persisted provider context. override
// must not be nil; component may be nil only for ScopeRoot.
func validateOverrideIdentity(
	override *nvidiacomv1beta1.ProviderOverride,
	scope Scope,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	// Re-resolve the registered target and reject stale or forged identities.
	expected, err := ExpectedTarget(consts.WorkloadProviderGrove, override.APIVersion, scope, component)
	if err != nil {
		return err
	}
	if override.Target != expected {
		return fmt.Errorf("target %q does not match resolved target %q", override.Target, expected)
	}

	// Recheck value ownership before the controller mutates provider resources.
	if valueErrs := ValidateValue(override.Target, override.Value.Raw); len(valueErrs) != 0 {
		return fmt.Errorf("value is invalid: %s", valueErrs[0].Error())
	}
	return nil
}

// patchNamedGroveTemplate merges a fragment into one named embedded target.
// result must not be nil.
func patchNamedGroveTemplate(
	result *unstructured.Unstructured,
	path []string,
	name string,
	patch []byte,
) error {
	// Read the generated list without assuming the destination exists.
	items, found, err := unstructured.NestedSlice(result.Object, path...)
	if err != nil {
		return fmt.Errorf("read %s: %w", strings.Join(path, "."), err)
	}
	if !found {
		return fmt.Errorf("generated destination %s[%q] was not found", strings.Join(path, "."), name)
	}

	// Merge only the generated entry whose stable name matches the DGD context.
	for i := range items {
		item, ok := items[i].(map[string]interface{})
		if !ok || item["name"] != name {
			continue
		}
		patched, err := mergeJSONObjects(item, patch)
		if err != nil {
			return fmt.Errorf("merge value into %s[%q]: %w", strings.Join(path, "."), name, err)
		}
		items[i] = patched
		return unstructured.SetNestedSlice(result.Object, items, path...)
	}
	return fmt.Errorf("generated destination %s[%q] was not found", strings.Join(path, "."), name)
}

func mergeJSONObjects(destination map[string]interface{}, patch []byte) (map[string]interface{}, error) {
	// Encode the rendered destination before applying JSON Merge Patch semantics.
	destinationJSON, err := json.Marshal(destination)
	if err != nil {
		return nil, err
	}

	// Apply the sparse provider fragment to the encoded destination.
	mergedJSON, err := jsonpatch.MergePatch(destinationJSON, patch)
	if err != nil {
		return nil, err
	}

	// Decode while preserving integer widths used by unstructured Kubernetes objects.
	var merged map[string]interface{}
	if err := sigsjson.UnmarshalCaseSensitivePreserveInts(mergedJSON, &merged); err != nil {
		return nil, err
	}
	if merged == nil {
		return nil, fmt.Errorf("merge result must be a JSON object")
	}
	return merged, nil
}
