window.addEventListener("load", e => {
    console.log("Loading input listeners!");

    initEventHandler("#input-grid-height", "change", elem => {
        let file = elem.files[0];
        if (!/^image/.test(file.type)) {
            console.log("Invalid input type for grid heights");
            return;
        }

        loadImage(URL.createObjectURL(file), (img) => {
            let w = img.width;
            let h = img.height;
            let grid = new Grid(w, h, img, (r, g, b) => 255 - r);
            loaded_grids["height"] = grid;
        });
    });

    initEventHandler("#input-grid-patch", "change", elem => {
        let file = elem.files[0];
        if (!/^image/.test(file.type)) {
            console.log("Invalid input type for grid patches");
            return;
        }

        loadImage(URL.createObjectURL(file), (img) => {
            let w = img.width;
            let h = img.height;
            let grid = new Grid(w, h, img, (r, g, b) => (r << 16) + (g << 8) + b);
            loaded_grids["patch"] = grid;
        });
    });

    initEventHandler("#input-grid-hierarchy", "change", elem => {
        let file = elem.files[0];
        if (!/^image/.test(file.type)) {
            console.log("Invalid input type for grid hierarchy");
            return;
        }

        loadImage(URL.createObjectURL(file), (img) => {
            let w = img.width;
            let h = img.height;
            let grid = new Grid(w, h, img, (r, g, b) => (r << 16) + (g << 8) + b);
            loaded_grids["hierarchy"] = grid;
        });
    });
    
    
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
});

const SCREEN_WIDTH_MAX = 600;
const SCREEN_HEIGHT_MAX = 600;
const SCREEN_WIDTH_MIN = 200;
const SCREEN_HEIGHT_MIN = 200;

let grid_to_show = null;
let current_grid = null;
let loaded_grids = {};

function setup() {
    let canvas = createCanvas(200, 200);
    canvas.parent("sketch-canvas");
    
    resizeSketch();
}

function draw() {
    current_grid = loaded_grids[grid_to_show];

    if (current_grid === null || current_grid === undefined) {
        background(204);
        return;
    }

    background(204);
    current_grid.render();

    let [pad_x, pad_y] = current_grid.padding();
    let cell_size = current_grid.cellSize();
    let cell_x = Math.floor((mouseX - pad_x) / cell_size);
    let cell_y = Math.floor((mouseY - pad_y) / cell_size);
    
    let inbounds = cell_x >= 0 && cell_x < current_grid.ncols && cell_y >= 0 && cell_y < current_grid.nrows;

    if (inbounds) {
        let value_height = loaded_grids["height"] ? loaded_grids["height"].dataAt(cell_x, cell_y) : "NA";
        let value_patch = loaded_grids["patch"] ? loaded_grids["patch"].dataAt(cell_x, cell_y) : "NA";
        let value_hierarchy = loaded_grids["hierarchy"] ? loaded_grids["hierarchy"].dataAt(cell_x, cell_y) : "NA";
        document.getElementById("display-cell-coords").innerHTML = "" + cell_x + ", " + cell_y;
        document.getElementById("display-cell-height").innerHTML = "" + value_height;
        document.getElementById("display-cell-patch").innerHTML = "" + value_patch;
        document.getElementById("display-cell-hierarchy").innerHTML = "" + value_hierarchy;
    }
}

function resizeSketch() {
    let sizeX = Math.max(SCREEN_WIDTH_MIN, Math.min(SCREEN_WIDTH_MAX, windowWidth));
    let sizeY = Math.max(SCREEN_HEIGHT_MIN, Math.min(SCREEN_HEIGHT_MAX, windowHeight));
    resizeCanvas(sizeX, sizeY);
}

function windowResized() {
    resizeSketch();
}

function initEventHandler(signature, eventName, callback) {
    let elem = typeof(signature) === "string"
        ? document.querySelector(signature)
        : signature;
    elem.addEventListener(eventName, event => callback(elem, event));
}


class Grid {
    constructor(ncols, nrows, img, mapping) {
        this.ncols = ncols;
        this.nrows = nrows;
        this.img = img;
        this.data = {};

        for (let i = 0; i < ncols; i++) {
            for (let j = 0; j < nrows; j++) {
                if (this.data[i] === undefined) {
                    this.data[i] = {};
                }
                let [r, g, b, a] = img.get(i, j);
                this.data[i][j] = mapping(r, g, b);
            }
        }
    }

    colorAt(x, y) {
        return this.img.get(x, y);
    }

    dataAt(x, y) {
        return this.data[x][y];
    }

    cellSize() {
        let grid_size = Math.max(this.ncols, this.nrows);
        return Math.min(width, height) / grid_size;
    }

    padding() {
        let pad_x, pad_y;
        let grid_size = Math.max(this.ncols, this.nrows);

        if (width > height) {
            pad_x = (width - height) / 2;
            pad_y = 0;
        }
        else {
            pad_x = 0;
            pad_y = (height - width) / 2;
        }

        if (this.ncols > this.nrows) {
            pad_y += ((grid_size - this.nrows) * this.cellSize()) / 2;
        }
        else if (this.nrows > this.ncols) {
            pad_x += ((grid_size - this.ncols) * this.cellSize()) / 2;
        }

        return [pad_x, pad_y];
    }

    render() {
        let [pad_x, pad_y] = this.padding();
        let cell_size = this.cellSize();

        for (let i = 0; i < this.ncols; i++) {
            for (let j = 0; j < this.nrows; j++) {
                push();
                let x = Math.floor(pad_x + i * cell_size);
                let y = Math.floor(pad_y + j * cell_size);
                // translate(x, y);
                // let [r, g, b] = this.getAt(i, j);
                let [r, g, b] = this.colorAt(i, j);
                fill(r, g, b);
                // fill(255, 0, 0);
                // stroke(0);
                noStroke();
                rect(x, y, cell_size, cell_size);
                pop();
            }
        }
    }
}
