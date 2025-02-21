import { app } from "../../scripts/app.js";

let ENABLE_WIDGET_HIDING = false; 
let RESDEBUG = false;
let ENABLE_UPDATED_TIMESTEP_SCALING = true;
let TOP_CLOWNDOG = true;
let DEFAULT_WIDGET_STATES = {};
const widgetToggleStates = new Map();

const originalGetNodeTypesCategories = typeof LiteGraph.getNodeTypesCategories === 'function' ? LiteGraph.getNodeTypesCategories : null;

// Override the getNodeTypesCategories method if it exists
if (originalGetNodeTypesCategories) {
    LiteGraph.getNodeTypesCategories = function(filter) {
        if (TOP_CLOWNDOG == false) {
            return originalGetNodeTypesCategories.call(this, filter);
        }
        
        try {
            // Get the original categories
            const categories = originalGetNodeTypesCategories.call(this, filter);
            
            categories.sort((a, b) => {
                const isARes4Lyf = a.startsWith("RES4LYF");
                const isBRes4Lyf = b.startsWith("RES4LYF");
                if (isARes4Lyf && !isBRes4Lyf) return -1;
                if (!isARes4Lyf && isBRes4Lyf) return 1;

                // Do the other auto sorting if enabled
                if (LiteGraph.auto_sort_node_types) {
                    return a.localeCompare(b);
                }
                return 0;
            });
            return categories;
        } catch (error) {
            return originalGetNodeTypesCategories.call(this, filter);
        }
    };
}

function resDebugLog(...args) {
    if (RESDEBUG) {
        console.log(...args);
        
        // Attempt to post the log text to the Python backend
        const logText = args.join(' ');
        fetch('/reslyf/log', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ log: logText })
        }).catch(error => {
            console.error('Error posting log to backend:', error);
        });
    }
}

const HIDDEN_TAG = "RES4LYF_hidden";
let origProps = {};
const nodeConfigs = {};

/**
 * Dynamic Widget Management Configuration
 * This system manages three types of widget visibility:
 * 1. Widget Dependencies: Widgets that show/hide based on other widgets' values
 * 2. Optional Input Dependencies: Widgets that show/hide based on input connections
 * 3. User-Set Hidden Widgets: Widgets that the user can choose to hide or show, with default states configurable in settings
**/

// 1. Widget Dependencies - These are common configurations that are shared across multiple nodes
const commonDependentWidgets = {
    groups: [
        {
            inputWidgetNames: ["noise_type", "noise_sampler_type"],     // Controlling widgets
            independentValues: ["fractal"],                             // Values that trigger showing dependent widgets
            widgetsToShow: ["alpha", "k"]                        // Widgets to show/hide
        },
        {
            inputWidgetNames: ["rk_type", "sampler_name"],
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
    "Legacy_ClownSampler",
    "ClownSampler",
    "ClownSamplerAdvanced",
    "ClownSamplerAdvanced_Beta",
    "ClownsharKSampler",
    "ClownsharKSamplerGuides",
    "ClownsharKSamplerGuide",
    //"ClownGuide_Beta",
    "ClownGuides_Beta",
    "ClownGuidesAB_Beta",
    "ClownGuidesFluxAdvanced_Beta",
    "Legacy_ClownSampler",
    "Legacy_SharkSampler",
    "Legacy_ClownsharKSampler",
    "SharkSampler",
    "UltraSharkSampler",
    "UltraSharkSampler Tiled",
];

// 2. Optional Input Dependencies - These are unique configurations for specific nodes
nodeConfigs["ClownsharKSamplerGuides"] = createNodeConfig("ClownsharKSamplerGuides", {
    optionalInputWidgets: {
        // Define groups of widgets that should be shown/hidden based on input connections
        groups: [
            {
                inputs: ["guide",],   // Show widgets if ANY of these inputs are connected
                widgets: ["guide_weight", "guide_weight_bkg", "guide_weight_scale", "guide_weight_scheduler", "guide_end_step"]  // Widgets to show/hide
            },
            {
                inputs: ["guide_bkg"],
                widgets: ["guide_weight_bkg", "guide_weight_bkg_scale", "guide_weight_scheduler_bkg", "guide_bkg_end_step"]
            }
        ]
    }
});

nodeConfigs["Legacy_ClownSampler"] = createNodeConfig("Legacy_ClownSampler", {
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

nodeConfigs["Legacy_ClownsharKSampler"] = createNodeConfig("Legacy_ClownsharKSampler", {
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

// 3. User-Set Hidden Widgets - These are widgets that the user can choose to hide or show
const TOGGLEABLE_WIDGETS = {
    "extra_options": {
        settingId: "RES4LYF.defaultHideExtraOptions",
        settingName: "RES4LYF: Hide extra_options widget by default",
        defaultValue: false
    },
    "denoise_alt": {
        settingId: "RES4LYF.defaultHide_denoise_alt",
        settingName: "RES4LYF: Hide denoise_alt widget by default",
        defaultValue: false
    },
};


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

    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
            origSerializeValue: widget.serializeValue
        };
    }

    widget._RES4LYF_hidden = !shouldShow;

    if (shouldShow) {
        showWidget(widget);
    } else {
        hideWidget(node, widget);
    }

    // Force node refresh so new computeSize is applied.
    const newSize = node.computeSize();
    node.setSize([node.size[0], Math.max(newSize[1], node.size[1])]);
    if (node.update) node.update();
    app.graph.setDirtyCanvas(true);
}

/**
 * Hides a widget
 * @param {object} node - The node object
 * @param {object} widget - The widget to hide
 */
function hideWidget(node, widget) {
    widget.type = "hidden";
    widget.computeSize = () => [0, 0];
    widget.serializeValue = () => null;
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.height = "0px";
        widget.element.style.overflow = "hidden";
    }
}

/**
 * Shows a widget
 * @param {object} widget - The widget to show
 */
function showWidget(widget) {
    widget.type = origProps[widget.name].origType;
    widget.computeSize = origProps[widget.name].origComputeSize;
    widget.serializeValue = origProps[widget.name].origSerializeValue;
    if (widget.element) {
        widget.element.style.display = "";
        widget.element.style.height = "";
        widget.element.style.overflow = "";
    }
}

/**
 * Creates a generic node change handler
 * @param {object} node - The node object
 * @param {array} dependentWidgetsConfig - Array of relevant widget names
 * @returns {function} - The node change handler
 */
function createGenericHandler(node, dependentWidgetsConfig) {
    return () => {
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

    // Initialize toggleable widgets with their default states
    Object.keys(TOGGLEABLE_WIDGETS).forEach(widgetName => {
        const widget = node.widgets.find(w => w.name === widgetName);
        if (widget) {
            const stateKey = `${node.id}_${widgetName}`;
            // Only apply default if user hasn't set a specific state
            if (!widgetToggleStates.has(stateKey)) {
                const defaultHidden = DEFAULT_WIDGET_STATES[widgetName];
                toggleWidget(node, widget, !defaultHidden);
            }
        }
    });

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

    // Add options to context menu
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
    
            // Add toggle options for each toggleable widget
            Object.keys(TOGGLEABLE_WIDGETS).forEach(widgetName => {
                const widget = node.widgets.find(w => w.name === widgetName);
                if (widget) {
                    const stateKey = `${node.id}_${widgetName}`;
                    const isHidden = widgetToggleStates.get(stateKey) ?? DEFAULT_WIDGET_STATES[widgetName];
    
                    options.push({
                        content: `${isHidden ? "Show" : "Hide"} ${widgetName}`,
                        callback: () => {
                            const newState = !isHidden;
                            widgetToggleStates.set(stateKey, newState);
                            toggleWidget(node, widget, !newState);
                            app.graph.setDirtyCanvas(true);
                        }
                    });
                }
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
    name: "Comfy.RES4LYF.DynamicWidgets",

    async setup(app) {
        Object.entries(TOGGLEABLE_WIDGETS).forEach(([widgetName, config]) => {
            app.ui.settings.addSetting({
                id: config.settingId,
                name: config.settingName,
                defaultValue: config.defaultValue,
                type: "boolean",
                options: [
                    { value: true, text: "Hidden" },
                    { value: false, text: "Shown" },
                ],
                onChange: (value) => {
                    DEFAULT_WIDGET_STATES[widgetName] = value;
                    resDebugLog(`Default state for ${widgetName} set to ${value ? "hidden" : "shown"}`);
                    
                    // Update all existing nodes
                    for (const node of app.graph._nodes) {
                        if (nodeConfigs[node.comfyClass]) {
                            const widget = node.widgets.find(w => w.name === widgetName);
                            if (widget) {
                                // Only update if user hasn't set a specific state for this instance
                                const stateKey = `${node.id}_${widgetName}`;
                                if (!widgetToggleStates.has(stateKey)) {
                                    toggleWidget(node, widget, !value);
                                }
                            }
                        }
                    }
                },
            });

            // Initialize default state
            DEFAULT_WIDGET_STATES[widgetName] = config.defaultValue;

        });

        app.ui.settings.addSetting({
            id: "RES4LYF.topClownDog",
            name: "RES4LYF: Top ClownDog",
            defaultValue: true,
            type: "boolean",
            options: [
                { value: true, text: "On" },
                { value: false, text: "Off" },
            ],
            onChange: (value) => {
                TOP_CLOWNDOG = value;
                resDebugLog(`Top ClownDog ${value ? "enabled" : "disabled"}`);
                
                // Send to backend
                fetch('/reslyf/settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        setting: "topClownDog",
                        value: value
                    })
                }).catch(error => {
                    resDebugLog(`Error updating topClownDog setting: ${error}`);
                });
            },
        });
        
        app.ui.settings.addSetting({
            id: "RES4LYF.enableDynamicWidgets",
            name: "RES4LYF: Enable dynamic widget hiding",
            defaultValue: false,
            type: "boolean",
            options: [
                { value: true, text: "On" },
                { value: false, text: "Off" },
            ],
            onChange: (value) => {
                ENABLE_WIDGET_HIDING = value;
                resDebugLog(`Dynamic widgets ${value ? "enabled" : "disabled"}`);
                
                // Send to backend
                fetch('/reslyf/settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        setting: "enableDynamicWidgets",
                        value: value
                    })
                }).catch(error => {
                    resDebugLog(`Error updating enableDynamicWidgets setting: ${error}`);
                });
        
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
            name: "RES4LYF: Enable JS debug logging",
            defaultValue: false,
            type: "boolean",
            options: [
                { value: true, text: "On" },
                { value: false, text: "Off" },
            ],
            onChange: (value) => {
                RESDEBUG = value;
                resDebugLog(`Debug logging ${value ? "enabled" : "disabled"}`);
                
                // Send to backend
                fetch('/reslyf/settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        setting: "enableDebugLogs",
                        value: value
                    })
                }).catch(error => {
                    resDebugLog(`Error updating enableDebugLogs setting: ${error}`);
                });
            },
        });
        
        const settingName = "RES4LYF.enableUpdatedTimestepScaling";
        app.ui.settings.addSetting({
            id: settingName,
            name: "RES4LYF (experimental): Enable \"improved\" timestep scaling for SD3.5 (May cause issues with other models eg. HYVideo)",
            defaultValue: false,
            type: "boolean",
            options: [
                { value: true, text: "On" },
                { value: false, text: "Off" },
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