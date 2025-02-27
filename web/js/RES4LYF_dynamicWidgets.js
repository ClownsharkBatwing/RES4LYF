import { app } from "../../scripts/app.js";

let RESDEBUG = false;
let TOP_CLOWNDOG = true;
let DISPLAY_CATEGORY = true;

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
        
        app.ui.settings.addSetting({
            id: "RES4LYF.displayCategory",
            name: "RES4LYF: Display Category in Sampler Names",
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

});