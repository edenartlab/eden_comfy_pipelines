from PIL import Image, ImageDraw

def create_bordered_image(width, height, border_fraction):
    # Calculate border width
    border = int(border_fraction * max(width, height))

    output_path=f"bordered_{border_fraction:.2f}_{max(width, height)}.png"

    # Create a white image
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Draw black rectangle border
    draw.rectangle([0, 0, width-1, height-1], outline="black", width=border)

    # Save result
    img.save(output_path)
    print(f"Saved bordered image to {output_path}")


if __name__ == "__main__":
    # Example usage
    create_bordered_image(1200, 800, 0.015)
