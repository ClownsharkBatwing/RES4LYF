import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

let RESDEBUG = false;
let TOP_CLOWNDOG = true;
let DISPLAY_CATEGORY = true;

let nodeCounter = 1;
const processedNodeMap = new WeakMap();

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

function debugLog(...args) {
    let force = false;
    if (typeof args[args.length - 1] === "boolean") {
        force = args.pop();
    }
    if (RESDEBUG || force) {
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

const resDebugLog = debugLog;

// Adapted from essentials.DisplayAny from ComfyUI_essentials
app.registerExtension({
    name: "Comfy.RES4LYF.DisplayInfo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("RES4LYF")) {
            return;
        }

        if (nodeData.name === "Latent Display State Info") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (this.widgets && this.widgets.length === 0) {
					for (let i = 1; i < this.widgets.length; i++) {
						this.widgets[i].onRemove?.();
					}
					this.widgets.length = 0;
				}

                // Check if the "text" widget already exists.
                let textWidget = this.widgets && this.widgets.length > 0 && this.widgets.find(w => w.name === "displaytext");
                if (!textWidget) {
                    textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", { multiline: true }], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.border = "none";
                    textWidget.inputEl.style.backgroundColor = "transparent";
                }
                textWidget.value = message["text"].join("");
            };
        }
    },
});

app.registerExtension({
    name: "Comfy.RES4LYF.DynamicWidgets",

    async setup(app) {
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
                debugLog(`Top ClownDog ${value ? "enabled" : "disabled"}`);
                
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
                    debugLog(`Error updating topClownDog setting: ${error}`);
                });
            },
        });
                
        app.ui.settings.addSetting({
            id: "RES4LYF.enableDebugLogs",
            name: "RES4LYF: Enable debug logging to console",
            defaultValue: false,
            type: "boolean",
            options: [
                { value: true, text: "On" },
                { value: false, text: "Off" },
            ],
            onChange: (value) => {
                RESDEBUG = value;
                debugLog(`Debug logging ${value ? "enabled" : "disabled"}`);
                
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
                    debugLog(`Error updating enableDebugLogs setting: ${error}`);
                });
            },
        });
        
        app.ui.settings.addSetting({
            id: "RES4LYF.displayCategory",
            name: "RES4LYF: Display Category in Sampler Names (requires browser refresh)",
            defaultValue: true,
            type: "boolean",
            options: [
                { value: true, text: "On" },
                { value: false, text: "Off" },
            ],
            onChange: (value) => {
                DISPLAY_CATEGORY = value;
                resDebugLog(`Display Category ${value ? "enabled" : "disabled"}`);
                
                // Send to backend
                fetch('/reslyf/settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        setting: "displayCategory",
                        value: value
                    })
                }).catch(error => {
                    resDebugLog(`Error updating displayCategory setting: ${error}`);
                });
            },
        });
        
    },


    nodeCreated(node) {
        if (NODES_WITH_EXPANDABLE_OPTIONS.includes(node.comfyClass)) {
            //debugLog(`Setting up expandable options for ${node.comfyClass}`, true);
            setupExpandableOptions(node);
        }
    }
});

const NODES_WITH_EXPANDABLE_OPTIONS = [
    "ClownsharKSampler_Beta",
    "ClownsharkChainsampler_Beta",
    "SharkChainsampler_Beta",


    "ClownSampler_Beta",
    "ClownSamplerAdvanced_Beta",

    "SharkSampler",
    "SharkSampler_Beta",
    "SharkSamplerAdvanced_Beta",

    "ClownOptions_Combine",
]

function setupExpandableOptions(node) {
    if (!processedNodeMap.has(node)) {
        processedNodeMap.set(node, ++nodeCounter);
        //debugLog(`Assigned ID ${nodeCounter} to node ${node.comfyClass}`);
    } else {
        //debugLog(`Node ${node.comfyClass} already processed with ID ${processedNodeMap.get(node)} - skipping`);
        return;
    }
        
    const originalOnConnectionsChange = node.onConnectionsChange;
    
    const hasOptionsInput = node.inputs.some(input => input.name === "options");
    if (!hasOptionsInput) {
        //debugLog(`Node ${node.comfyClass} doesn't have an options input - skipping`);
        return;
    }
    
    node.onConnectionsChange = function(type, index, connected, link_info) {
        if (originalOnConnectionsChange) {
            originalOnConnectionsChange.call(this, type, index, connected, link_info);
        }
        
        if (type === LiteGraph.INPUT && !connected) {
            const input = this.inputs[index];
            if (!input || !input.name.startsWith("options")) {
                return;
            }
            
            //debugLog(`Options input disconnected: ${input.name}`);
            
            // setTimeout to let the graph update first
            setTimeout(() => {
                cleanupOptionsInputs(this);
            }, 100);
            return;
        }
        
        if (type === LiteGraph.INPUT && connected && link_info) {
            const input = this.inputs[index];
            if (!input || !input.name.startsWith("options")) {
                return;
            }
            
            let hasEmptyOptions = false;
            for (let i = 0; i < this.inputs.length; i++) {
                const input = this.inputs[i];
                if (input.name.startsWith("options") && input.link === null) {
                    hasEmptyOptions = true;
                    break;
                }
            }
            
            if (!hasEmptyOptions) {
                //debugLog(`All options inputs are connected, adding a new one`);
                
                // Find the highest index number in existing options inputs
                let maxIndex = 0;
                for (let i = 0; i < this.inputs.length; i++) {
                    const input = this.inputs[i];
                    if (input.name === "options") {
                        continue; // Skip the base "options" input
                    } else if (input.name.startsWith("options ")) {
                        const match = input.name.match(/options (\d+)/);
                        if (match) {
                            const index = parseInt(match[1]) - 1;
                            maxIndex = Math.max(maxIndex, index);
                        }
                    }
                }
                
                const newName = maxIndex === 0 ? "options 2" : `options ${maxIndex + 2}`;
                this.addInput(newName, "OPTIONS");
                //debugLog(`Created new options input: ${newName}`);
                
                this.setDirtyCanvas(true, true);
            }
        }
    };

    const optionsInputs = node.inputs.filter(input => 
        input.name.startsWith("options")
    );
    
    const baseOptionsInput = optionsInputs.find(input => input.name === "options");
    const hasOptionsWithIndex = optionsInputs.some(input => input.name !== "options");
    
    // if (baseOptionsInput && !hasOptionsWithIndex) {
    //     debugLog(`Adding initial options 1 input to ${node.comfyClass}`);
    //     node.addInput("options 1", "OPTIONS");
    //     node.setDirtyCanvas(true, true);
    // }
    
    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function(info) {
        if (originalOnConfigure) {
            originalOnConfigure.call(this, info);
        }
        
        let hasEmptyOptions = false;
        for (let i = 0; i < this.inputs.length; i++) {
            const input = this.inputs[i];
            if (input.name.startsWith("options") && input.link === null) {
                hasEmptyOptions = true;
                break;
            }
        }
        
        if (!hasEmptyOptions && this.inputs.some(i => i.name.startsWith("options"))) {
            let maxIndex = 0;
            for (let i = 0; i < this.inputs.length; i++) {
                const input = this.inputs[i];
                if (input.name === "options") {
                    continue;
                } else if (input.name.startsWith("options ")) {
                    const match = input.name.match(/options (\d+)/);
                    if (match) {
                        const index = parseInt(match[1]) - 1;
                        maxIndex = Math.max(maxIndex, index);
                    }
                }
            }
            
            const newName = maxIndex === 0 ? "options 2" : `options ${maxIndex + 2}`;
            this.addInput(newName, "OPTIONS");
        }
    };

    function cleanupOptionsInputs(node) {
        const optionsInputs = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            if (input.name.startsWith("options")) {
                optionsInputs.push({
                    index: i,
                    name: input.name,
                    connected: input.link !== null,
                    isBase: input.name === "options"
                });
            }
        }
        
        const baseInput = optionsInputs.find(info => info.isBase);
        const nonBaseInputs = optionsInputs.filter(info => !info.isBase);
        
        let needsRenumbering = false;
        
        if (baseInput && !baseInput.connected && nonBaseInputs.every(info => !info.connected)) {
            nonBaseInputs.sort((a, b) => b.index - a.index);
            
            for (const inputInfo of nonBaseInputs) {
                //debugLog(`Removing unnecessary options input: ${inputInfo.name} (index ${inputInfo.index})`);
                node.removeInput(inputInfo.index);
                needsRenumbering = true;
            }
            
            node.setDirtyCanvas(true, true);
            return;
        }
        
        const disconnectedInputs = nonBaseInputs.filter(info => !info.connected);
        
        if (disconnectedInputs.length > 1) {
            disconnectedInputs.sort((a, b) => b.index - a.index);
            
            for (let i = 1; i < disconnectedInputs.length; i++) {
                //debugLog(`Removing unnecessary options input: ${disconnectedInputs[i].name} (index ${disconnectedInputs[i].index})`);
                node.removeInput(disconnectedInputs[i].index);
                needsRenumbering = true;
            }
        }
        
        const hasConnectedOptions = optionsInputs.some(info => info.connected);
        const hasEmptyOptions = optionsInputs.some(info => !info.connected && !info.isBase);
        
        if (hasConnectedOptions && !hasEmptyOptions) {
            node.addInput("options temp", "OPTIONS");
            //debugLog(`Added new empty options input`);
            needsRenumbering = true;
        }
        
        if (needsRenumbering) {
            renumberOptionsInputs(node);
            node.setDirtyCanvas(true, true);
        }
    }
    
    function renumberOptionsInputs(node) {
        const optionsInfo = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            if (input.name.startsWith("options")) {
                if (input.name === "options") {
                    continue;
                }
                
                optionsInfo.push({
                    index: i,
                    connected: input.link !== null,
                    name: input.name
                });
            }
        }
        
        optionsInfo.sort((a, b) => {
            if (a.connected !== b.connected) {
                return b.connected ? 1 : -1; // Connected inputs first
            }
            return a.index - b.index;
        });
        
        for (let i = 0; i < optionsInfo.length; i++) {
            const inputInfo = optionsInfo[i];
            const newName = `options ${i + 2}`;
            
            if (inputInfo.name !== newName) {
                //debugLog(`Renaming ${inputInfo.name} to ${newName}`);
                node.inputs[inputInfo.index].name = newName;
            }
        }
    }
}
