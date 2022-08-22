const SCREEN_WIDTH_MAX = 600;
const SCREEN_HEIGHT_MAX = 600;
const SCREEN_WIDTH_MIN = 200;
const SCREEN_HEIGHT_MIN = 200;

let grid_to_show = null;
let current_grid = null;
let loaded_grids = {};
let selected_cell = null;

function setup() {
    let canvas = createCanvas(200, 200);
    canvas.parent("sketch-canvas");
    noSmooth();
    
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
        document.getElementById("display-mouse-cell-coords").innerHTML = "" + cell_x + ", " + cell_y;
        document.getElementById("display-mouse-cell-height").innerHTML = "" + value_height;
        document.getElementById("display-mouse-cell-patch").innerHTML = "" + value_patch;
        document.getElementById("display-mouse-cell-hierarchy").innerHTML = "" + value_hierarchy;
    }
    else {
        document.getElementById("display-mouse-cell-coords").innerHTML = "NA";
        document.getElementById("display-mouse-cell-height").innerHTML = "NA";
        document.getElementById("display-mouse-cell-patch").innerHTML = "NA";
        document.getElementById("display-mouse-cell-hierarchy").innerHTML = "NA";
    }

    if (selected_cell !== null) {
        push();
        stroke(255, 0, 0);
        strokeWeight(2);
        noFill();
        rect(pad_x + selected_cell[0] * cell_size, pad_y + selected_cell[1] * cell_size, cell_size, cell_size);
        pop();
    }
}

function mousePressed() {
    if (current_grid === null || current_grid === undefined) {
        return;
    }

    let [pad_x, pad_y] = current_grid.padding();
    let cell_size = current_grid.cellSize();
    let cell_x = Math.floor((mouseX - pad_x) / cell_size);
    let cell_y = Math.floor((mouseY - pad_y) / cell_size);
    
    let inbounds = cell_x >= 0 && cell_x < current_grid.ncols && cell_y >= 0 && cell_y < current_grid.nrows;
    let same_cell = selected_cell !== null && selected_cell[0] == cell_x && selected_cell[1] == cell_y;
    
    if (inbounds) {
        if (!same_cell) {
            selected_cell = [cell_x, cell_y];
            let value_height = loaded_grids["height"] ? loaded_grids["height"].dataAt(cell_x, cell_y) : "NA";
            let value_patch = loaded_grids["patch"] ? loaded_grids["patch"].dataAt(cell_x, cell_y) : "NA";
            let value_hierarchy = loaded_grids["hierarchy"] ? loaded_grids["hierarchy"].dataAt(cell_x, cell_y) : "NA";
            document.getElementById("display-selected-cell-coords").innerHTML = "" + cell_x + ", " + cell_y;
            document.getElementById("display-selected-cell-height").innerHTML = "" + value_height;
            document.getElementById("display-selected-cell-patch").innerHTML = "" + value_patch;
            document.getElementById("display-selected-cell-hierarchy").innerHTML = "" + value_hierarchy;
        }
        else {
            selected_cell = null;
            document.getElementById("display-selected-cell-coords").innerHTML = "NA";
            document.getElementById("display-selected-cell-height").innerHTML = "NA";
            document.getElementById("display-selected-cell-patch").innerHTML = "NA";
            document.getElementById("display-selected-cell-hierarchy").innerHTML = "NA";
        }
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

