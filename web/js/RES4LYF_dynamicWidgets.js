import { app } from "../../scripts/app.js";

let RESDEBUG = true;
let ENABLE_WIDGET_HIDING = true;

function resDebugLog(...args) {
    if (RESDEBUG) {
        console.log(...args);
    }
}

// Constant used to mark hidden widgets
const HIDDEN_TAG = "RES4LYF_hidden";

// Object to store original widget properties
let origProps = {};

const nodeConfigs = {
    "AdvancedNoise": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "ClownSamplerLegacy": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "ClownSampler": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "KSamplerSelectAdvanced": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "LatentNoised": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerCorona": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerDEIS_SDE": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerDPMPP_2M_SDE_Advanced": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerDPMPP_2S_Ancestral_Advanced": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerDPMPP_3M_SDE_Advanced": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerDPMPP_DualSDE_Advanced": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerDPMPP_SDE_Advanced": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerDPMPP_SDE_CFG++_Advanced": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerEulerAncestral_Advanced": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerNoiseInversion": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerRES_Implicit": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerRES3_Implicit_Automation": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerRK": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SamplerRK_Test": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "SharkSampler": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "UltraSharkSampler": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    "UltraSharkSampler Tiled": {
        relevantWidgets: ["noise_type", "noise_sampler_type"],
    },
    
};

/**
 * Toggles the visibility of a widget
 * @param {object} node - The node object
 * @param {object} widget - The widget to toggle
 * @param {boolean} show - Whether to show or hide the widget
 */
function toggleWidget(node, widget, show = false) {
    if (!widget || !ENABLE_WIDGET_HIDING) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }
    const origSize = node.size;

    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    if (show)
        delete widget.computedHeight;
    else
        widget.computedHeight = 0;

    const newSize = node.computeSize();
    const height = show ? Math.max(newSize[1], origSize[1]) : newSize[1];
    node.setSize([node.size[0], height]);
}

/**
 * Creates a generic node change handler
 * @param {object} node - The node object
 * @param {array} relevantWidgets - Array of relevant widget names
 * @returns {function} - The node change handler
 */
function createGenericHandler(node, relevantWidgets) {
    return () => {
        resDebugLog(`${node.comfyClass} onNodeChange called`);
        const noiseTypeWidget = node.widgets.find(w => w.name === "noise_type");
        const noiseSamplerTypeWidget = node.widgets.find(w => w.name === "noise_sampler_type");
        const alphaWidget = node.widgets.find(w => w.name === "alpha");
        const kWidget = node.widgets.find(w => w.name === "k");

        if (noiseTypeWidget && alphaWidget && kWidget) {
            const showFractalInputs = noiseTypeWidget.value === "fractal";
            toggleWidget(node, alphaWidget, showFractalInputs);
            toggleWidget(node, kWidget, showFractalInputs);
        }

        if (noiseSamplerTypeWidget && alphaWidget && kWidget) {
            const showFractalInputs = noiseSamplerTypeWidget.value === "fractal";
            toggleWidget(node, alphaWidget, showFractalInputs);
            toggleWidget(node, kWidget, showFractalInputs);
        }
    };
}

/**
 * Sets up dynamic widgets for a node
 * @param {object} node - The node object
 * @param {array} relevantWidgets - Array of relevant widget names
 */
function setupDynamicWidgets(node, relevantWidgets) {
    resDebugLog("Setting up dynamic widgets for", node.comfyClass);
    resDebugLog("Relevant widgets:", relevantWidgets);
    const onNodeChange = createGenericHandler(node, relevantWidgets);

    // Call onNodeChange initially
    onNodeChange();

    // Set up the callback for relevant widgets
    relevantWidgets.forEach(widgetName => {
        const widget = node.widgets.find(w => w.name === widgetName);
        if (widget) {
            resDebugLog(`Setting up ${widgetName} callback for ${node.comfyClass}`);
            widget.callback = onNodeChange;
        }
    });

    // Override the node's onConnectionsChange method
    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function(type, index, connected, link_info) {
        if (origOnConnectionsChange) {
            origOnConnectionsChange.apply(this, arguments);
        }
        onNodeChange();
    };

    // Add custom menu option
    const getExtraMenuOptions = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function(_, options) {
        const result = getExtraMenuOptions ? getExtraMenuOptions.apply(this, arguments) : undefined;
        options.push({
            content: "Refresh Dynamic Widgets",
            callback: onNodeChange
        });
        return result;
    };
}

app.registerExtension({
    name: "Comfy.RES4LYF.DynamicWidgets",
    nodeCreated(node) {
        resDebugLog("Node created:", node.comfyClass);

        const config = nodeConfigs[node.comfyClass];
        if (config) {
            resDebugLog(`${node.comfyClass} node detected`);

            // Pass config.relevantWidgets to setupDynamicWidgets
            setupDynamicWidgets(node, config.relevantWidgets);

            // Override onConfigure
            const onConfigure = node.onConfigure;
            node.onConfigure = function(o) {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                // Pass config.relevantWidgets here as well
                setupDynamicWidgets(node, config.relevantWidgets);
                return r;
            }

            // Add onAfterGraphConfigured
            node.onAfterGraphConfigured = function () {
                requestAnimationFrame(() => {
                    const handler = createGenericHandler(node, config.relevantWidgets);
                    handler();
                });
            };
        }
    },
});