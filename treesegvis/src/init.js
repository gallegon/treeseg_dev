function initEventHandler(signature, eventName, callback) {
    let elem = typeof(signature) === "string"
        ? document.querySelector(signature)
        : signature;
    // console.log("Init: " + signature);
    elem.addEventListener(eventName, event => callback(elem, event));
}

function initFileInput(grid_type, data_decoder, display_decoder) {
    initEventHandler("#input-grid-" + grid_type, "change", elem => {
        let file = elem.files[0];
        if (!/^image/.test(file.type)) {
            console.log("Invalid input type for grid (expected image)");
            return;
        }

        loadImage(URL.createObjectURL(file), (img) => {
            let w = img.width;
            let h = img.height;
            let grid = new Grid(w, h, img, data_decoder, display_decoder);
            loaded_grids[grid_type] = grid;
            let radio = document.getElementById("input-show-grid-" + grid_type);
            radio.checked = true;
            radio.dispatchEvent(new Event("change"));
        });
    });
}

function initSketchInterface() {
    // console.log("Loading input listeners!");

    initFileInput("height", (r, g, b) => 255 - r, img => img);
    initFileInput("patch", (r, g, b) => (r << 16) + (g << 8) + b, (img, data) => createDisplayImage(data));
    initFileInput("hierarchy", (r, g, b) => (r << 16) + (g << 8) + b, (img, data) => createDisplayImage(data));    
    
    document.getElementsByName("input-grid-to-show").forEach(radio => {
        initEventHandler(radio, "change", elem => {
            switch (elem.id) {
                case "input-show-grid-height":
                    grid_to_show = "height";
                    break;
                case "input-show-grid-patch":
                    grid_to_show = "patch";
                    break;
                case "input-show-grid-hierarchy":
                    grid_to_show = "hierarchy";
                    break;
                default:
                    break;
            }
        });
    });

    let status = document.getElementById("display-treeseg-status");
    initEventHandler("#input-treeseg-run", "click", () => {
        // console.log("Clicked!");
        status.innerHTML = "Running..."
        fetch("/treeseg-run", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: document.getElementById("input-treeseg-params").value
        })
        .then(data => data.json())
        .then(data => {
            // console.log(data);
            let elapsed = data["elapsed_time"]
            status.innerHTML = "Finished in " + elapsed + " seconds"

            let src_height = "data:image/png;base64," + data["data-grid-height"]
            loadImage(src_height, img => {
                // console.log("Loaded an image!");
                // console.log(img.width + " x " + img.height);
                let w = img.width;
                let h = img.height;
                let grid = new Grid(w, h, img, (r, g, b) => 255 - r, img => img);
                loaded_grids["height"] = grid;
                mask_hierarchy = null;
            });

            let src_patch = "data:image/png;base64," + data["data-grid-patch"]
            loadImage(src_patch, img => {
                // console.log("Loaded an image!");
                // console.log(img.width + " x " + img.height);
                let w = img.width;
                let h = img.height;
                let grid = new Grid(w, h, img, (r, g, b) => (r << 16) + (g << 8) + b, (img, data) => createDisplayImage(data));
                loaded_grids["patch"] = grid;
                mask_hierarchy = null;
            });

            let src_hierarchy = "data:image/png;base64," + data["data-grid-hierarchy"]
            loadImage(src_hierarchy, img => {
                // console.log("Loaded an image!");
                // console.log(img.width + " x " + img.height);
                let w = img.width;
                let h = img.height;
                let grid = new Grid(w, h, img, (r, g, b) => (r << 16) + (g << 8) + b, (img, data) => createDisplayImage(data));
                loaded_grids["hierarchy"] = grid;
                mask_hierarchy = null;
                let radio = document.getElementById("input-show-grid-hierarchy");
                radio.checked = true;
                radio.dispatchEvent(new Event("change"));
            });
        });
    });
}

window.addEventListener("load", initSketchInterface);
