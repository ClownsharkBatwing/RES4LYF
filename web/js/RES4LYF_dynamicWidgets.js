import { app } from "../../scripts/app.js";

let ENABLE_WIDGET_HIDING = false; 
let RESDEBUG = false;
let ENABLE_UPDATED_TIMESTEP_SCALING = true;

function resDebugLog(...args) {
    if (RESDEBUG) {
        console.log(...args);
    }
}

const HIDDEN_TAG = "RES4LYF_hidden";
let origProps = {};
const nodeConfigs = {};

/**
 * Dynamic Widget Management Configuration
 * This system manages two types of widget visibility:
 * 1. Widget Dependencies: Widgets that show/hide based on other widgets' values
 * 2. Optional Input Dependencies: Widgets that show/hide based on input connections
 */

// 1. Widget Dependencies - These are common configurations that are shared across multiple nodes
const commonDependentWidgets = {
    groups: [
        {
            inputWidgetNames: ["noise_type", "noise_sampler_type"],     // Controlling widgets
            independentValues: ["fractal"],                             // Values that trigger showing dependent widgets
            widgetsToShow: ["alpha", "k"]                        // Widgets to show/hide
        },
        {
            inputWidgetNames: ["rk_type"],
            independentValues: [
                "dormand-prince_6s", 
                "dormand-prince_7s", 
                "dormand-prince_13s",
                "bogacki-shampine_7s",
                "rk_exp_5s", 
                "rk4_4s", 
                "rk38_4s", 
                "ralston_4s", 
                "dpmpp_3s", 
                "heun_3s", 
                "houwen-wray_3s", 
                "kutta_3s", 
                "ralston_3s", 
                "res_3s", 
                "ssprk3_3s", 
                "dpmpp_2s", 
                "dpmpp_sde_2s", 
                "heun_2s", 
                "ralston_2s", 
                "res_2s"
            ],
            widgetsToShow: ["multistep"]
        }
    ]
};

// Names of nodes that will use commonDependentWidgets config
const nodesWithCommonConfig = [
    "AdvancedNoise",
    "ClownSampler",
    "ClownSamplerLegacy",
    "KSamplerSelectAdvanced",
    "LatentNoised",
    "SamplerCorona",
    "SamplerDEIS_SDE",
    "SamplerDPMPP_2M_SDE_Advanced",
    "SamplerDPMPP_2S_Ancestral_Advanced",
    "SamplerDPMPP_3M_SDE_Advanced",
    "SamplerDPMPP_DualSDE_Advanced",
    "SamplerDPMPP_SDE_Advanced",
    "SamplerDPMPP_SDE_CFG++_Advanced",
    "SamplerEulerAncestral_Advanced",
    "SamplerNoiseInversion",
    "SamplerRES_Implicit",
    "SamplerRES3_Implicit_Automation",
    "SamplerRK",
    "SamplerRK_Test",
    "SharkSampler",
    "UltraSharkSampler",
    "UltraSharkSampler Tiled"
];

// 2. Optional Input Dependencies - These are unique configurations for specific nodes
nodeConfigs["ClownSampler"] = createNodeConfig("ClownSampler", {
    optionalInputWidgets: {
        // Define groups of widgets that should be shown/hidden based on input connections
        groups: [
            {
                inputs: ["latent_guide", "latent_guide_inv"],   // Show widgets if ANY of these inputs are connected
                widgets: ["latent_guide_weight", "guide_mode"]  // Widgets to show/hide
            },
            {
                inputs: ["latent_guide_mask"],
                widgets: ["rescale_floor"]
            }
        ]
    }
});

nodeConfigs["SamplerRK"] = createNodeConfig("SamplerRK", {
    optionalInputWidgets: {
        groups: [
            {
                inputs: ["latent_guide", "latent_guide_inv"],
                widgets: ["latent_guide_weight", "guide_mode"]
            }
        ]
    }
});

// Function to create node configurations
function createNodeConfig(nodeName, uniqueConfig = {}) {
    const config = { ...uniqueConfig };

    if (nodesWithCommonConfig.includes(nodeName)) {
        config.dependentWidgets = commonDependentWidgets;
    }

    return config;
}

// Add remaining nodes with common configurations but no unique configurations
nodesWithCommonConfig.forEach(nodeName => {
    if (!nodeConfigs[nodeName]) {
        nodeConfigs[nodeName] = createNodeConfig(nodeName);
    }
});

/**
 * Toggles the visibility of a widget
 * @param {object} node - The node object
 * @param {object} widget - The widget to toggle
 * @param {boolean} shouldShow - Whether the widget should be shown
 */
function toggleWidget(node, widget, shouldShow = false) {
    if (!widget) return;

    resDebugLog(`Toggling widget ${widget.name} in ${node.comfyClass}: shouldShow=${shouldShow}, ENABLE_WIDGET_HIDING=${ENABLE_WIDGET_HIDING}`);
    
    if (!origProps[widget.name]) {
        origProps[widget.name] = { 
            origType: widget.type,
            origComputeSize: widget.computeSize 
        };
    }
    
    const origSize = node.size;

    widget._RES4LYF_hidden = !shouldShow;
    widget.type = shouldShow || !ENABLE_WIDGET_HIDING ? origProps[widget.name].origType : HIDDEN_TAG;
    widget.computeSize = shouldShow || !ENABLE_WIDGET_HIDING ? 
        origProps[widget.name].origComputeSize : 
        () => [0, -4];

    if (shouldShow || !ENABLE_WIDGET_HIDING)
        delete widget.computedHeight;
    else
        widget.computedHeight = 0;

    const newSize = node.computeSize();
    const height = shouldShow || !ENABLE_WIDGET_HIDING ? 
        Math.max(newSize[1], origSize[1]) : 
        newSize[1];
    node.setSize([node.size[0], height]);
}

/**
 * Creates a generic node change handler
 * @param {object} node - The node object
 * @param {array} dependentWidgetsConfig - Array of relevant widget names
 * @returns {function} - The node change handler
 */
function createGenericHandler(node, dependentWidgetsConfig) {
    return () => {
        resDebugLog(`${node.comfyClass} onNodeChange called`);

        if (dependentWidgetsConfig?.groups) {
            dependentWidgetsConfig.groups.forEach(group => {
                const { inputWidgetNames, independentValues, widgetsToShow } = group;

                const showWidgets = inputWidgetNames.some(widgetName => {
                    const widget = node.widgets.find(w => w.name === widgetName);
                    return widget && independentValues.includes(widget.value);
                });

                widgetsToShow.forEach(widgetName => {
                    const widget = node.widgets.find(w => w.name === widgetName);
                    toggleWidget(node, widget, showWidgets);
                });
            });
        }
    };
}

/**
 * Sets up dynamic widgets for a node
 * @param {object} node - The node object
 * @param {array} dependentWidgets - Array of relevant widget names
 */
function setupDynamicWidgets(node, dependentWidgetsConfig) {
    resDebugLog("Setting up dynamic widgets for", node.comfyClass);
    const onNodeChange = createGenericHandler(node, dependentWidgetsConfig);

    onNodeChange();

    if (dependentWidgetsConfig?.groups) {
        dependentWidgetsConfig.groups.forEach(group => {
            group.inputWidgetNames.forEach(widgetName => {
                const widget = node.widgets.find(w => w.name === widgetName);
                if (widget) {
                    widget.callback = onNodeChange;
                }
            });
        });
    }

    // Connection change handler
    function handleConnectionChange(type, index, connected, link_info) {
        const config = nodeConfigs[node.comfyClass];
        if (!config?.optionalInputWidgets?.groups) return;
    
        // Find which input changed
        const inputName = node.inputs[index]?.name;
        if (!inputName) return;
    
        // Check each group
        config.optionalInputWidgets.groups.forEach(group => {
            if (group.inputs.includes(inputName)) {
                // Check if ANY input in the group is connected
                const isAnyConnected = group.inputs.some(inputName => {
                    const input = node.inputs.find(input => input.name === inputName);
                    return input && input.link !== null;
                });
    
                // Toggle all widgets in the group
                group.widgets.forEach(widgetName => {
                    const widget = node.widgets.find(w => w.name === widgetName);
                    if (widget) {
                        toggleWidget(node, widget, isAnyConnected);
                    }
                });
            }
        });
    }
    
    // Override the node's onConnectionsChange method
    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function(type, index, connected, link_info) {
        if (origOnConnectionsChange) {
            origOnConnectionsChange.apply(this, arguments);
        }
        handleConnectionChange(type, index, connected, link_info);
        onNodeChange();
    };

    // Add refresh option to context menu
    if (!node._RES4LYF_menuWrapped) {
        const originalGetExtraMenuOptions = node.getExtraMenuOptions;
        node.getExtraMenuOptions = function(_, options) {
            if (originalGetExtraMenuOptions) {
                originalGetExtraMenuOptions.apply(this, arguments);
            }
            options.push({
                content: "Refresh RES4LYF widgets",
                callback: onNodeChange
            });
        };
        node._RES4LYF_menuWrapped = true;
    }
    
    // Initialize optional input widgets' visibility
    const config = nodeConfigs[node.comfyClass];
    if (config?.optionalInputWidgets?.groups) {
        config.optionalInputWidgets.groups.forEach(group => {
            // Check if ANY input in the group is connected
            const isAnyConnected = group.inputs.some(inputName => {
                const input = node.inputs.find(input => input.name === inputName);
                return input && input.link !== null;
            });
    
            // Set initial visibility for all widgets in the group
            group.widgets.forEach(widgetName => {
                const widget = node.widgets.find(w => w.name === widgetName);
                if (widget) {
                    toggleWidget(node, widget, isAnyConnected);
                }
            });
        });
    }
}

app.registerExtension({
    async setup(app) {
        app.ui.settings.addSetting({
            id: "RES4LYF.enableDynamicWidgets",
            name: "RES4LYF: Enable dynamic widget hiding",
            defaultValue: false,
            type: "boolean",
            options: (value) => [
                { value: true, text: "On", selected: value === true },
                { value: false, text: "Off", selected: value === false },
            ],
            onChange: (value) => {
                ENABLE_WIDGET_HIDING = value;
                resDebugLog(`Dynamic widgets ${value ? "enabled" : "disabled"}`);
                
                for (const node of app.graph._nodes) {
                    if (!nodeConfigs[node.comfyClass]) {
                        resDebugLog(`Skipping node ${node.comfyClass} - not in config`);
                        continue;
                    }
                    
                    resDebugLog(`Processing node ${node.comfyClass}`);
                    node.widgets.forEach(widget => {
                        if (widget._RES4LYF_hidden !== undefined) {
                            toggleWidget(node, widget, !widget._RES4LYF_hidden);
                        }
                    });
                }
            },
        });
    
        app.ui.settings.addSetting({
            id: "RES4LYF.enableDebugLogs",
            name: "RES4LYF: Enable debug logging",
            defaultValue: false,
            type: "boolean",
            options: (value) => [
                { value: true, text: "On", selected: value === true },
                { value: false, text: "Off", selected: value === false },
            ],
            onChange: (value) => {
                RESDEBUG = value;
                resDebugLog(`Debug logging ${value ? "enabled" : "disabled"}`);
            },
        });

        app.ui.settings.addSetting({
            id: "RES4LYF.enableUpdatedTimestepScaling",
            name: "RES4LYF: Enable \"improved\" timestep scaling",
            defaultValue: true,
            type: "boolean",
            options: (value) => [
                { value: true, text: "On", selected: value === true },
                { value: false, text: "Off", selected: value === false },
            ],
            onChange: (value) => {
                // Send to backend
                fetch('/reslyf/settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        setting: "updatedTimestepScaling",
                        value: value
                    })
                }).then(response => {
                    if (response.ok) {
                        ENABLE_UPDATED_TIMESTEP_SCALING = value;
                        resDebugLog(`Updated timestep scaling ${value ? "enabled" : "disabled"}`);
                    } else {
                        resDebugLog(`Failed to update updatedTimestepScaling setting`);
                    }
                }).catch(error => {
                    resDebugLog(`Error updating updatedTimestepScaling setting: ${error}`);
                });
            }
        });
    },

    name: "Comfy.RES4LYF.DynamicWidgets",
    nodeCreated(node) {
        resDebugLog("Node created:", node.comfyClass);

        const config = nodeConfigs[node.comfyClass];
        if (config) {
            resDebugLog(`${node.comfyClass} node detected`);

            // Set up dynamic widgets
            setupDynamicWidgets(node, config.dependentWidgets);

            // Refresh widgets immediately
            const onNodeChange = createGenericHandler(node, config.dependentWidgets);
            onNodeChange();

            // Handle configuration
            const onConfigure = node.onConfigure;
            node.onConfigure = function(o) {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                setupDynamicWidgets(node, config.dependentWidgets);
                return r;
            }

            // Handle graph configuration
            node.onAfterGraphConfigured = function () {
                requestAnimationFrame(() => {
                    if (node && node.graph) {  // Check if node is still valid
                        const handler = createGenericHandler(node, config.dependentWidgets);
                        handler();
                    }
                });
            };
        }
    },
});