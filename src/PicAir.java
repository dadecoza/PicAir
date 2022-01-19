import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.core.*;

public class PicAir {

	private List<Point> points = new ArrayList<Point>();

	BufferedImage mat2Image(Mat videoMatImage) {
		int type = videoMatImage.channels() == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;
		int bufferSize = videoMatImage.channels() * videoMatImage.cols() * videoMatImage.rows();
		byte[] buffer = new byte[bufferSize];
		videoMatImage.get(0, 0, buffer);
		BufferedImage image = new BufferedImage(videoMatImage.cols(), videoMatImage.rows(), type);
		final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
		return image;
	}

	public Mat whiteDetection(Mat hsvImage) {
		Mat threshedImg = new Mat();
		Scalar hsvMin = new Scalar(0, 0, 255);
		Scalar hsvMax = new Scalar(255, 10, 255);
		Core.inRange(hsvImage, hsvMin, hsvMax, threshedImg);
		return threshedImg;
	}

	public Mat greenDetection(Mat hsvImage) {
		Mat threshedImg = new Mat();
		Scalar hsvMin = new Scalar(45, 0, 0);
		Scalar hsvMax = new Scalar(85, 255, 255);
		Core.inRange(hsvImage, hsvMin, hsvMax, threshedImg);
		return threshedImg;
	}

	public Mat removeNoise(Mat frame) {
		Mat blurredImage = new Mat();
		Mat hsvImage = new Mat();
		Imgproc.blur(frame, blurredImage, new Size(7, 7));
		Imgproc.cvtColor(blurredImage, hsvImage, Imgproc.COLOR_BGR2HSV);
		return hsvImage;
	}

	public Mat morph(Mat frame) {
		Mat morphOutput = new Mat();
		Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6));
		Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
		Imgproc.erode(frame, morphOutput, erodeElement);
		Imgproc.erode(morphOutput, morphOutput, erodeElement);
		Imgproc.dilate(morphOutput, morphOutput, dilateElement);
		Imgproc.dilate(morphOutput, morphOutput, dilateElement);
		return morphOutput;
	}

	public Point getPoint(Mat frame) {
		Mat nonoise = removeNoise(frame);
		Mat morphed = morph(nonoise);
		Mat white = whiteDetection(morphed);
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Mat hierarchy = new Mat();
		Imgproc.findContours(white, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		for (MatOfPoint cont : contours) {
			Rect rect = Imgproc.boundingRect(cont);

			int padding = 20;
			int x = rect.x - padding;
			int y = rect.y - padding;
			int w = rect.width + padding * 2;
			int h = rect.height + padding * 2;

			try {
				Rect roi = new Rect(x, y, w, h);
				Mat rectCrop = frame.submat(roi);
				if (isGreen(rectCrop)) {
					double centerX = rect.x + 0.5 * rect.width;
					double centerY = rect.y + 0.5 * rect.height;
					return new Point(centerX, centerY);
				}
			} catch (CvException e) {
				// new rect probably out of bounds
			}
		}
		return null;
	}

	public boolean isGreen(Mat frame) {
		Mat nonoise = removeNoise(frame);
		Mat morphed = morph(nonoise);
		Mat green = greenDetection(morphed);
		Mat hierarchy = new Mat();
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(green, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		return contours.size() > 0;
	}

	PicAir() {

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		Mat frame = new Mat();
		// 0; default video device id
		VideoCapture camera = new VideoCapture(0);
		JFrame jframe = new JFrame("Air Draw");
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		JLabel vidpanel = new JLabel();
		jframe.setContentPane(vidpanel);
		jframe.setVisible(true);
		boolean packed = false;
		while (true) {
			if (camera.read(frame)) {

				Mat flipped = new Mat();
				Core.flip(frame, flipped, 1);

				Point point = getPoint(flipped);

				if (point != null) {
					this.points.add(point);
				}

				for (int i = 1; i < points.size(); i++) {
					Point a = points.get(i);
					Point b = points.get(i - 1);
					double dist = Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2);
					if (dist < 500) {
						Imgproc.line(flipped, a, b, new Scalar(0, 255, 0), 3);
					}
				}

				ImageIcon image = new ImageIcon(mat2Image(flipped));
				vidpanel.setIcon(image);
				vidpanel.repaint();

				if (!packed) {
					jframe.pack();
					packed = true;
				}
			}
		}
	}

	public static void main(String[] args) {
		new PicAir();
	}
}