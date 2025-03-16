import { app } from "../../scripts/app.js";

// For debugging - confirm the script is loaded
console.log("Eden NumberDisplay extension loading...");

app.registerExtension({
    name: "Eden.NumberDisplay",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only modify Eden_RandomNumberSampler nodes
        if (nodeData.name !== "Eden_RandomNumberSampler") {
            return;
        }
        
        console.log("Registering Eden_RandomNumberSampler node definition");
        
        // Store the original onDrawForeground function
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        
        // Override the onDrawForeground function to display the sampled value
        nodeType.prototype.onDrawForeground = function(ctx) {
            // Call the original onDrawForeground function if it exists
            if (onDrawForeground) {
                onDrawForeground.apply(this, arguments);
            }
            
            // Check if we have a value to display
            if (this.sampledValue) {
                // Draw a background for the text
                const textHeight = 20;
                const margin = 10;
                const width = this.size[0];
                const y = this.size[1] - textHeight - margin/2;
                
                ctx.fillStyle = "rgba(0,0,0,0.2)";
                ctx.beginPath();
                ctx.roundRect(0, y, width, textHeight, [0, 0, 5, 5]);
                ctx.fill();
                
                // Draw the text
                ctx.fillStyle = "#FFF";
                ctx.font = "14px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Sampled: " + this.sampledValue, width/2, y + 15);
            }
        };
        
        // Store the original onExecuted function
        const onExecuted = nodeType.prototype.onExecuted;
        
        // Override the onExecuted function
        nodeType.prototype.onExecuted = function(message) {
            // Call the original onExecuted function if it exists
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }
            
            console.log("Eden_RandomNumberSampler executed with message:", message);
            
            // Store the random number to display
            if (message && message.random_number !== undefined) {
                this.sampledValue = message.random_number[0];
                // Force the node to redraw
                this.setDirtyCanvas(true, true);
            }
        };
    }
});

console.log("Eden NumberDisplay extension loaded successfully");