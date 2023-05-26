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

    def find_features_patron(self, image_A, image_B) -> tuple:
        circle_rows = 9
        circle_cols = 16
        ret, pts1 = cv2.findCirclesGrid(cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY), (circle_cols, circle_rows), None)
        ret, pts2 = cv2.findCirclesGrid(cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY), (circle_cols, circle_rows), None)
        return pts1.reshape(-1, 2), pts2.reshape(-1, 2)

    def __call__(self, enable_bundle_adjustment:boolean= False):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        pose_array = self.img_obj.K.ravel() # Aplanamiento de la matriz K en una matriz unidimensional
        transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))

        pose_A = np.matmul(self.img_obj.K, transform_matrix_0) # Calcula la matriz de pose de la cámara, que se utiliza para convertir los puntos de la imagen coordenadas del mundo real
        total_points = np.zeros((1, 3)) # total_points almacena todos los puntos 3D estimados durante el proceso

        image_A = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_B = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        features_AB, features_BA = self.find_features_patron(image_A, image_B) # Coincidencia de puntos clave en ambas imágenes y calcular las características distintivas de la imagen

        essential_matrix, em_mask = cv2.findEssentialMat(features_AB, features_BA, self.img_obj.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None) # Calcular la matriz esencial que representa la relación geométrica entre las dos vistas

        _, R, t, em_mask = cv2.recoverPose(essential_matrix, features_AB, features_BA, self.img_obj.K) # Recuperamos la rotación (R) y la traslación (t) de la cámara a partir de la matriz esencial y los puntos característicos correspondientes en las dos vistas de la escena

        # Actualizamos transform_matrix_1 para incluir la rotación y al traslación de la segunda cámara
        transform_matrix_1[:3, :3] = np.matmul(R, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], t.ravel())

        pose_B = np.matmul(self.img_obj.K, transform_matrix_1)

        # TRIANGULACIÓN ESTÉREO
        pt_cloud = cv2.triangulatePoints(pose_A, pose_B, features_AB.T, features_BA.T) # points_3d es una matriz de forma (4, N), donde N es el número de puntos característicos.
        points_3d = (pt_cloud / pt_cloud[3]) # Se divide cada fila de pt_cloud por el último elemento (pt_cloud[3]) para obtener una matriz de forma (3, N) que contiene las coordenadas 3D en el espacio del mundo real.
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)


        _, _, _, inlier = cv2.solvePnPRansac(points_3d[:, 0, :], features_BA, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)  # En las dos primeras vistas solamente usamos esta función para eliminar outliers
        if inlier is not None: features_BA, points_3d = features_BA[inlier[:, 0]], points_3d[:, 0, :][inlier[:, 0]]

        pose_array = np.hstack((np.hstack((pose_array, pose_A.ravel())), pose_B.ravel())) # pose_array almacena las matrices de proyección de todas las cámaras
        puntos_ancla = np.copy(points_3d)
        for i in tqdm(range(len(self.img_obj.image_list) - 2 )):
            image_C = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
            features_BC, features_CB = self.find_features_patron(image_B, image_C)

            _, rvec, t, inlier = cv2.solvePnPRansac(puntos_ancla, features_CB, self.img_obj.K, np.zeros((5, 1), dtype=np.float32),cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(rvec)

            # Obtenemos las matrices de transformación y proyección de la imagen C
            transform_matrix_1 = np.hstack((R, t))
            pose_C = np.matmul(self.img_obj.K, transform_matrix_1)

            points_3d = cv2.triangulatePoints(pose_B, pose_C, features_BC.T, features_CB.T)
            points_3d = points_3d/points_3d[3]
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)


            pose_array = np.hstack((pose_array, pose_C.ravel()))

            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(points_3d, features_CB, transform_matrix_1, self.img_obj.K, 0.5)
                pose_C = np.matmul(self.img_obj.K, transform_matrix_1)
                total_points = np.vstack((total_points, points_3d))
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))

            image_B = np.copy(image_C)
            pose_B = np.copy(pose_C)
            cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_C)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        print("Printing to .ply file")
        self.to_ply(self.img_obj.path, total_points, np.zeros((total_points.shape[0], 3)))
        np.savetxt(self.img_obj.path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2]+'_pose_array.csv', pose_array, delimiter = '\n')

if __name__ == '__main__':
    sfm = Sfm("Datasets\\patron")
    sfm()

