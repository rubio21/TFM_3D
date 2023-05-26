import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm
import matplotlib.pyplot as plt


class Image_loader():
    def __init__(self, img_dir: str, downscale_factor: float):
        with open(img_dir + '\\K.txt') as f:
            self.K = np.array(list((map(lambda x: list(map(lambda x: float(x), x.strip().split(' '))), f.read().split('\n')))))
            self.image_list = []
        for image in sorted(os.listdir(img_dir)):
            if image[-4:].lower() == '.jpg' or image[-4:].lower() == '.png':
                self.image_list.append(img_dir + '\\' + image)
        self.path = os.getcwd()
        self.factor = downscale_factor
        self.downscale()

    def downscale(self) -> None:
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor

    def downscale_image(self, image):
        for _ in range(1, int(self.factor / 2) + 1):
            image = cv2.pyrDown(image)
        return image


class Sfm():
    def __init__(self, img_dir: str, downscale_factor: float = 1.0) -> None:
        self.img_obj = Image_loader(img_dir, downscale_factor)

    def reprojection_error(self, obj_points, image_points, transform_matrix, K, homogenity) ->tuple:
        R = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(R)
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points

    def optimal_reprojection_error(self, obj_points) -> np.array:
        '''
        calculates of the reprojection error during bundle adjustment
        returns error 
        '''
        transform_matrix = obj_points[0:12].reshape((3,4))
        K = obj_points[12:21].reshape((3,3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3), 3))
        R = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(R)
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]
        error = [ (p[idx] - image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel()/len(p)

    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

        values_corrected = least_squares(self.optimal_reprojection_error, opt_variables, gtol = r_error).x
        K = values_corrected[12:21].reshape((3,3))
        rest = int(len(values_corrected[21:]) * 0.4)
        return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:])/3), 3)), values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, values_corrected[0:12].reshape((3,4))

    def to_ply(self, path, point_cloud, colors) -> None:
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])
        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2] + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')


    def common_points(self, features_BA, features_BC, features_CB) -> tuple:
        cm_points_0 = []
        cm_points_1 = []

        for i in range(features_BA.shape[0]):
            a = np.where(features_BC == features_BA[i, :])
            if a[0].size != 0:
                cm_points_0.append(i)
                cm_points_1.append(a[0][0])

        mask_array_1 = np.ma.array(features_BC, mask=False)
        mask_array_1.mask[cm_points_1] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(features_CB, mask=False)
        mask_array_2.mask[cm_points_1] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_0), np.array(cm_points_1), mask_array_1.T, mask_array_2.T

    def find_features(self, image_A, image_B) -> tuple:
        sift = cv2.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY), None)

        index_params, search_params = dict(algorithm=1, trees=2), dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_0, desc_1, k=2)
        matchesMask = [[0, 0] for i in range(len(matches))]
        pts1, pts2, good = [], [], []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.55 * n.distance:
                matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(key_points_1[m.trainIdx].pt)
                pts1.append(key_points_0[m.queryIdx].pt)
        # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=cv2.DrawMatchesFlags_DEFAULT)
        # keypoint_matches = cv2.drawMatchesKnn(image_A, key_points_0, image_B, key_points_1, matches, None, **draw_params)
        # cv2.imshow("Keypoint_matches", keypoint_matches)
        # cv2.waitKey()
        return np.float32(pts1), np.float32(pts2)

    def __call__(self, enable_bundle_adjustment:boolean= False):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        pose_array = self.img_obj.K.ravel() # Aplanamiento de la matriz K en una matriz unidimensional
        transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))

        pose_A = np.matmul(self.img_obj.K, transform_matrix_0) # Calcula la matriz de pose de la cámara, que se utiliza para convertir los puntos de la imagen coordenadas del mundo real
        total_points = np.zeros((1, 3)) # total_points almacena todos los puntos 3D estimados durante el proceso
        total_colors = np.zeros((1, 3)) # total_colors almacena todos los colores de los puntos 3D estimados durante el proceso

        image_A = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_B = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        features_AB, features_BA = self.find_features(image_A, image_B) # Coincidencia de puntos clave en ambas imágenes y calcular las características distintivas de la imagen

        essential_matrix, em_mask = cv2.findEssentialMat(features_AB, features_BA, self.img_obj.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None) # Calcular la matriz esencial que representa la relación geométrica entre las dos vistas
        features_AB, features_BA = features_AB[em_mask.ravel() == 1], features_BA[em_mask.ravel() == 1]

        _, R, t, em_mask = cv2.recoverPose(essential_matrix, features_AB, features_BA, self.img_obj.K) # Recuperamos la rotación (R) y la traslación (t) de la cámara a partir de la matriz esencial y los puntos característicos correspondientes en las dos vistas de la escena
        features_AB, features_BA = features_AB[em_mask.ravel() > 0], features_BA[em_mask.ravel() > 0]

        # Actualizamos transform_matrix_1 para incluir la rotación y al traslación de la segunda cámara
        transform_matrix_1[:3, :3] = np.matmul(R, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], t.ravel())

        pose_B = np.matmul(self.img_obj.K, transform_matrix_1)

        # TRIANGULACIÓN ESTÉREO
        pt_cloud = cv2.triangulatePoints(pose_A, pose_B, features_AB.T, features_BA.T) # points_3d es una matriz de forma (4, N), donde N es el número de puntos característicos.
        points_3d = (pt_cloud / pt_cloud[3]) # Se divide cada fila de pt_cloud por el último elemento (pt_cloud[3]) para obtener una matriz de forma (3, N) que contiene las coordenadas 3D en el espacio del mundo real.

        error, points_3d = self.reprojection_error(points_3d, features_BA.T, transform_matrix_1, self.img_obj.K, homogenity = 1) # El error de reproyección es la diferencia entre los puntos 2D de la imagen y los puntos 2D proyectados en la imagen a partir de las coordenadas 3D estimadas en la triangulación estéreo.
        print("REPROJECTION ERROR: ", error)

        _, _, _, inlier = cv2.solvePnPRansac(points_3d[:, 0, :], features_BA, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)  # En las dos primeras vistas solamente usamos esta función para eliminar outliers
        if inlier is not None: features_BA, points_3d = features_BA[inlier[:, 0]], points_3d[:, 0, :][inlier[:, 0]]

        pose_array = np.hstack((np.hstack((pose_array, pose_A.ravel())), pose_B.ravel())) # pose_array almacena las matrices de proyección de todas las cámaras
        for i in tqdm(range(len(self.img_obj.image_list) - 2 )):
            image_C = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))

            features_BC, features_CB = self.find_features(image_B, image_C) #features_BA y features_BC son características de la misma imagen, pero features_BA es del matching con la imagen anterior y features_BC es del matching con la próxima imagen que vamos a usar en esta iteración

            if i != 0:
                # TRIANGULACIÓN ESTÉREO
                pt_cloud = cv2.triangulatePoints(pose_A, pose_B, features_AB.T, features_BA.T)  # pt_cloud es una matriz de forma (4, N), donde N es el número de puntos característicos.
                points_3d = pt_cloud / pt_cloud[3]  # Se divide cada fila de pt_cloud por el último elemento (pt_cloud[3]) para obtener una matriz de forma (3, N) que contiene las coordenadas 3D en el espacio del mundo real.
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]

            # indexes_features_BA contiene los índices de features_BA de los puntos coincidentes entre features_BA y features_BC. indexes_features_BC contiene los indices de features_BC de los puntos coincidentes entre features_BA y features_BC.
            # features_BC_minus_common_BA_and_BC es un subconjunto de features_BC que excluye los puntos comunes encontrados entre features_BA y features_BC
            # features_CB_minus_common_BA_and_BC es un subconjunto de features_CB que excluye los puntos comunes encontrados entre features_BA y features_BC.
            indexes_features_BA, indexes_features_BC, features_BC_minus_common_BA_and_BC, features_CB_minus_common_BA_and_BC = self.common_points(features_BA, features_BC, features_CB)
            features_CB_intersection_features_BA = features_CB[indexes_features_BC] # features_CB_intersection_features_BA contiene los puntos de features_CB coincidentes con features_BA.

            # Calculamos la rotación y la translación de la cámara. Como puntos 3D le pasamos los puntos de la imagen B que también están en A. Como puntos 2D le pasamos los puntos de C que también están en A. De esta manera estimamos la rotación y la traslación de la imagen C
            _, rvec, t, inlier = cv2.solvePnPRansac(points_3d[indexes_features_BA], features_CB_intersection_features_BA, self.img_obj.K, np.zeros((5, 1), dtype=np.float32),cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(rvec)

            # Obtenemos las matrices de transformación y proyección de la imagen C
            transform_matrix_1 = np.hstack((R, t))
            pose_C = np.matmul(self.img_obj.K, transform_matrix_1)

            # Obtenemos los puntos 3D haciendo la triangulación con las matrices de proyección de la imagen B y de la imagen C
            points_3d = cv2.triangulatePoints(pose_B, pose_C, features_BC_minus_common_BA_and_BC, features_CB_minus_common_BA_and_BC)
            points_3d = points_3d/points_3d[3]

            error, points_3d = self.reprojection_error(points_3d, features_CB_minus_common_BA_and_BC, transform_matrix_1, self.img_obj.K, homogenity = 1)
            print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, pose_C.ravel()))

            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(points_3d, features_CB_minus_common_BA_and_BC, transform_matrix_1, self.img_obj.K, 0.5)
                pose_C = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 0)
                print("Bundle Adjusted error: ",error)
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_C[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(features_CB_minus_common_BA_and_BC, dtype=np.int32)
                color_vector = np.array([image_C[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector))

            pose_A = np.copy(pose_B)
            plt.scatter(i, error)
            plt.pause(0.05)

            image_B = np.copy(image_C)
            features_AB = np.copy(features_BC)
            features_BA = np.copy(features_CB)
            pose_B = np.copy(pose_C)
            cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_C)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()

        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.img_obj.path, total_points, total_colors)
        print("Completed Exiting ...")
        np.savetxt(self.img_obj.path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')

if __name__ == '__main__':
    sfm = Sfm("Datasets\\sin_rectificar_color")
    sfm()

