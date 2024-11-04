import pdb
import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        folder_path = path
        image_files = sorted(glob.glob(folder_path + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(image_files)))

        if len(image_files) < 2:
            print("Not enough images to stitch.")
            return None, []

        homography_matrices = []

        base_image = cv2.imread(image_files[0])
        for idx in range(1, len(image_files)):
            next_image = cv2.imread(image_files[idx])

            key_pts1, desc1, key_pts2, desc2 = self.get_keypoint(base_image, next_image)
            matches = self.match_keypoint(key_pts1, key_pts2, desc1, desc2)

            match_points = np.array([[key_pts1[m.queryIdx].pt[0], key_pts1[m.queryIdx].pt[1], 
                                      key_pts2[m.trainIdx].pt[0], key_pts2[m.trainIdx].pt[1]]
                                     for m in matches])

            best_homography = self.ransac(match_points)

            if best_homography is None:

                del next_image
                continue

            homography_matrices.append(best_homography)
            base_image = self.stitch_images(base_image, next_image, best_homography)

            del next_image

        print("Stitching completed.")
        return base_image, homography_matrices

    def get_keypoint(self, base_image, next_image):
        sift = cv2.SIFT_create()
        key_pts1, desc1 = sift.detectAndCompute(base_image, None)
        key_pts2, desc2 = sift.detectAndCompute(next_image, None)

        

        return key_pts1, desc1, key_pts2, desc2

    def match_keypoint(self, key_pts1, key_pts2, desc1, desc2):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        knn_matches = flann.knnMatch(desc1, desc2, k=2)
        good_matches = []

        for m, n in knn_matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)


        return good_matches

    def ransac(self, match_points):
        best_inlier_set = []
        best_homography = None
        threshold = 5
        for _ in range(10):
            sampled_pts = random.sample(match_points.tolist(), k=4)
            H = self.homography(sampled_pts)
            inliers = []
            for pt in match_points:
                p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
                p_prime = np.array([pt[2], pt[3], 1]).reshape(3, 1)
                Hp = np.dot(H, p)
                Hp /= Hp[2]
                dist = np.linalg.norm(p_prime - Hp)

                if dist < threshold:
                    inliers.append(pt)

            if len(inliers) > len(best_inlier_set):
                best_inlier_set, best_homography = inliers, H

        return best_homography

    def homography(self, points):
        A = []
        for pt in points:
            x, y = pt[0], pt[1]
            X, Y = pt[2], pt[3]
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

        A = np.array(A)
        _, _, vh = np.linalg.svd(A)
        H = (vh[-1, :].reshape(3, 3))
        H /= H[2, 2]
        return H

    def stitch_images(self, base_image, next_image, best_homography):
        rows1, cols1 = next_image.shape[:2]
        rows2, cols2 = base_image.shape[:2]

        pts1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

        transformed_pts = cv2.perspectiveTransform(pts2, best_homography)
        combined_pts = np.concatenate((pts1, transformed_pts), axis=0)

        [x_min, y_min] = np.int32(combined_pts.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(combined_pts.max(axis=0).ravel() + 0.5)

        translation_H = (np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]])).dot(best_homography)

        output_img = cv2.warpPerspective(base_image, translation_H, (x_max - x_min, y_max - y_min))
        output_img[(-y_min):rows1 + (-y_min), (-x_min):cols1 + (-x_min)] = next_image

        del base_image

        return output_img

    def say_hi(self):
        print('Hey you ! From Shashank Ghosh..')
