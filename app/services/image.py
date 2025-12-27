from PIL import Image, ImageDraw, ImageFont
import io

class ImageService:
    @staticmethod
    def draw_bounding_boxes(image_bytes: bytes, boxes: list) -> bytes:
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        w, h = image.size
        
        for box in boxes:
            # Gemini standard: ymin, xmin, ymax, xmax
            ymin, xmin, ymax, xmax = box
            
            # Y coordinates map to Height, X coordinates map to Width
            left = (xmin / 1000) * w
            top = (ymin / 1000) * h
            right = (xmax / 1000) * w
            bottom = (ymax / 1000) * h
            
            draw.rectangle([left, top, right, bottom], outline="#FF0000", width=4)
            
        # ... save and return buffer ... tokens i kept 4096 if its okay then go with this token limit
        output_buffer = io.BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(output_buffer, format="JPEG")
        return output_buffer.getvalue()

image_service = ImageService()