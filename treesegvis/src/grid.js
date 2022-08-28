function decodeImage(img, decoder) {
    let w = img.width;
    let h = img.height;
    let decoded = [];
    for (let i = 0; i < w; i++) {
        decoded[i] = [];
    }
    
    img.loadPixels();
    for (let i = 0; i < w; i++) {
        for (let j = 0; j < h; j++) {
            let [r, g, b, a] = img.get(i, j);
            let val = decoder(r, g, b);
            decoded[i][j] = val;
        }
    }

    console.log("Decoded an image")
    console.log(decoded);
    return decoded;
}

function createDisplayImage(data) {
    let palette = {};
    let w = data.length;
    let h = data[0].length;
    let disp_img = createImage(w, h);
    let col_min = 20;
    let col_max = 235;
    let next_r = Math.floor(random(col_min, col_max));
    let next_g = Math.floor(random(col_min, col_max));
    let next_b = Math.floor(random(col_min, col_max));

    disp_img.loadPixels();
    for (let i = 0; i < w; i++) {
        for (let j = 0; j < h; j++) {
            let value = data[i][j];
            if (value == 0) {
                disp_img.set(i, j, 0);
                continue;
            }

            if (!(value in palette)) {
                palette[value] = [next_r, next_g, next_b];
                next_r = Math.floor(random(col_min, col_max));
                next_g = Math.floor(random(col_min, col_max));
                next_b = Math.floor(random(col_min, col_max));
            }

            let [r, g, b] = palette[value];
            disp_img.set(i, j, color(r, g, b));
        }
    }
    disp_img.updatePixels();
    return disp_img;
}

function createImageMaskFromData(data, match) {
    let w = data.length;
    let h = data[0].length;
    let mask = createImage(w, h);
    let mask_color = color(0, 255, 0, 127);
    mask.loadPixels();
    for (let i = 0; i < w; i++) {
        for (let j = 0; j < h; j++) {
            let value = data[i][j];
            if (value == match) {
                mask.set(i, j, mask_color);
            }
            else {
                mask.set(i, j, color(0, 0, 0, 0));
            }
        }
    }
    mask.updatePixels();
    return mask;
}

class Grid {
    constructor(ncols, nrows, img, data_decoder, display_decoder) {
        this.ncols = ncols;
        this.nrows = nrows;
        this.data = decodeImage(img, data_decoder);
        this.display_image = display_decoder(img, this.data);
    }

    colorAt(x, y) {
        return this.display_img.get(x, y);
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

        image(this.display_image, pad_x, pad_y, this.ncols * cell_size, this.nrows * cell_size);
    }
}
