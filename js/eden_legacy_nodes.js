import { app } from "../../scripts/app.js";

// Core ComfyUI took over the "SaveImageAdvanced" node id. Old Eden workflows store
// [filename_prefix, add_timestamp, save_metadata_json]; core stores [filename_prefix, format, ...].
function migrateNodes(nodes) {
    for (const node of nodes ?? []) {
        if (node.type === "SaveImageAdvanced" && typeof node.widgets_values?.[1] === "boolean") {
            node.type = "Eden_SaveImageAdvanced";
        }
    }
}

app.registerExtension({
    name: "Eden.LegacyNodeIds",
    beforeConfigureGraph(graphData) {
        migrateNodes(graphData?.nodes);
        for (const subgraph of graphData?.definitions?.subgraphs ?? []) {
            migrateNodes(subgraph.nodes);
        }
    },
});
