package br.com.x10d.reconhecimento;

import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;

public class A_TesteJavaCV {
    public static void main(String args[]) {
        FaceRecognizer r = EigenFaceRecognizer.create();
    }
}
