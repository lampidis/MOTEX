import cv2
import numpy as np
import matplotlib.pyplot as plt

class LaneDetector:
    def __init__(self,
                 sobel_thresh=(20, 255),
                 hls_white_thresh=(0, 200, 0, 255, 255, 255),
                 hls_yellow_thresh=(15, 30, 115, 35, 204, 255),
                 hough_params=dict(rho=1, theta=np.pi/180, threshold=50,
                                   minLineLength=40, maxLineGap=100),
                 roi_vertices_ratio=[(0.05, 0.9), (0.25, 0.4),
                                     (0.85, 0.4), (0.95, 0.9)],
                 angle_thresh=(20, 60)):
        """
        Initialize the LaneDetector with threshold parameters and ROI shape.
        sobel_thresh: tuple(min, max) for gradient thresholding
        hls_white_thresh: tuple(h_min, l_min, s_min, h_max, l_max, s_max)
        hls_yellow_thresh: same as above for yellow
        hough_params: dict of parameters for cv2.HoughLinesP
        roi_vertices_ratio: list of 4 (x_ratio, y_ratio) vertices defining a trapezoid
        """
        self.sobel_thresh = sobel_thresh
        self.hls_white_thresh = hls_white_thresh
        self.hls_yellow_thresh = hls_yellow_thresh
        self.hough_params = hough_params
        self.roi_vertices_ratio = roi_vertices_ratio
        self.angle_thresh = angle_thresh

    def color_threshold(self, image):
        """Apply HLS color threshold to isolate white and yellow lanes."""
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        # unpack thresholds
        wh_lo = np.array(self.hls_white_thresh[:3], dtype=np.uint8)
        wh_hi = np.array(self.hls_white_thresh[3:], dtype=np.uint8)
        yl_lo = np.array(self.hls_yellow_thresh[:3], dtype=np.uint8)
        yl_hi = np.array(self.hls_yellow_thresh[3:], dtype=np.uint8)

        white_mask = cv2.inRange(hls, wh_lo, wh_hi)
        yellow_mask = cv2.inRange(hls, yl_lo, yl_hi)
        return cv2.bitwise_or(white_mask, yellow_mask)

    def gradient_threshold(self, image):
        """Apply Sobel X gradient thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel = np.absolute(sobelx)
        scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        low, high = self.sobel_thresh
        _, binary = cv2.threshold(scaled, low, high, cv2.THRESH_BINARY)
        return binary

    def combine_masks(self, color_mask, grad_mask):
        """Combine color and gradient masks."""
        return cv2.bitwise_or(color_mask, grad_mask)

    def region_of_interest(self, mask):
        """Mask everything outside the trapezoidal ROI."""
        h, w = mask.shape
        vertices = np.array([[(int(x * w), int(y * h))
                              for (x, y) in self.roi_vertices_ratio]], np.int32)
        roi_mask = np.zeros_like(mask)
        cv2.fillPoly(roi_mask, vertices, 255)
        return cv2.bitwise_and(mask, roi_mask)

    def hough_lines(self, masked):
        """Detect line segments using Hough Transform."""
        return cv2.HoughLinesP(masked, **self.hough_params)

    def average_slope_intercept(self, lines, height, width):
        left, right = [], []
        if lines is None:
            return None, None
        min_ang, max_ang = self.angle_thresh
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            angle = abs(np.degrees(np.arctan(slope)))
            # filter by angle
            if angle < min_ang or angle > max_ang:
                continue
            intercept = y1 - slope * x1
            x_bottom = (height - intercept) / slope
            # skip if bottom intersection is outside frame
            if x_bottom < 0 or x_bottom > width:
                continue
            if slope < 0:
                left.append((slope, intercept))
            else:
                right.append((slope, intercept))

        def make_line(avg):
            slope, intercept = avg
            y1 = height
            y2 = int(height * self.roi_vertices_ratio[1][1])
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return (x1, y1, x2, y2)

        left_line = make_line(np.mean(left, axis=0)) if left else None
        right_line = make_line(np.mean(right, axis=0)) if right else None
        return left_line, right_line

    def draw_lines(self, image, lines):
        """Draw lines onto a blank image."""
        line_img = np.zeros_like(image)
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)
        return line_img

    def detect(self, image):
        """Full pipeline: returns original image overlaid with detected lanes."""
        c_mask = self.color_threshold(image)
        # g_mask = self.gradient_threshold(image)
        # combined = self.combine_masks(c_mask, g_mask)
        roi = self.region_of_interest(c_mask)
        lines = self.hough_lines(roi)
        left, right = self.average_slope_intercept(lines, image.shape[0], image.shape[1])
        overlay = self.draw_lines(image, [left, right])
        result = cv2.addWeighted(image, 0.8, overlay, 1, 0)
        return result
    # {
    #             "original": image,
    #             "color_mask": c_mask,
    #             "gradient_mask": g_mask,
    #             "combined_mask": combined,
    #             "roi_mask": roi,
    #             "final": result
    #         }

if __name__ == '__main__':
    # Example usage:
    img_path = '../TwinLiteNet/images/cb22c820-f094952f.jpg'  # or other image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    detector = LaneDetector()
    steps = detector.detect(img)

    # Plot in a 2x3 grid
    titles = ["Original", "Color Mask", "Gradient Mask", "Combined Mask", "ROI Mask", "Final Output"]
    plt.figure(figsize=(15, 10))
    for idx, key in enumerate(steps):
        plt.subplot(2, 3, idx+1)
        if key == "original" or key == "final":
            plt.imshow(cv2.cvtColor(steps[key], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(steps[key], cmap='gray')
        plt.title(titles[idx])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
