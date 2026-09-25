import { app } from "../../scripts/app.js";

// Shows the last sampled value of Eden_RandomNumberSampler at the bottom of the node.
app.registerExtension({
    name: "Eden.NumberDisplay",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Eden_RandomNumberSampler") return;

        const LABEL_SPACE = 28;
        const computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const size = computeSize.apply(this, arguments);
            size[1] += LABEL_SPACE;
            return size;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.sampledValue === undefined || this.flags?.collapsed) return;

            const height = 20;
            const width = this.size[0];
            const y = this.size[1] - height - 5;

            ctx.save();
            ctx.fillStyle = "rgba(0,0,0,0.2)";
            ctx.beginPath();
            ctx.roundRect(0, y, width, height, [0, 0, 5, 5]);
            ctx.fill();
            ctx.fillStyle = "#FFF";
            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Sampled: " + this.sampledValue, width / 2, y + 15);
            ctx.restore();
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const value = message?.random_number?.[0];
            if (value !== undefined) {
                this.sampledValue = value;
                const minSize = this.computeSize();
                if (this.size[1] < minSize[1]) this.setSize([this.size[0], minSize[1]]);
                this.setDirtyCanvas(true, true);
            }
        };
    },
});
