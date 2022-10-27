from ast import BitXor
from distutils.command.check import HAS_DOCUTILS
import numpy as np
import argparse
import cv2
import math
import time 
import tqdm

class PerspectiveTransformCalibration():
    def __init__(self):
        self.imgsz_init = np.array([2464,2056])
        self.imgsz = np.array([320,320])
        self.factor = self.imgsz_init/self.imgsz

    

    def getIP_WP(self, cam):
        if cam == "haunter":
            IP_1 = [0.2736691457537848*2464,0.2952503209242619*2056]
            IP_2 = [0.5148267020238813*2464,0.27086007702182285*2056]
            # if args.cheat :
            #     IP_2 = [0.5148267020238813*2464,0.27086007702182285*2090]
            IP_3 = [0.30475167522859725*2464,0.34403080872913994*2056]
            IP_4 = [0.571632704167504*2464,0.3196405648267009*2056]

            WP_1 = [0.0,0.0]
            WP_2 = [2.0,0.0]
            WP_3 = [0.0,2.0]
            WP_4 = [2.0,2.0]
        elif cam == "staging":
            IP_1 = [953, 1551]
            IP_2 = [745, 1823]
            IP_3 = [1825,1613]
            IP_4 = [1729,1891]

            WP_1 = [0.0,0.0]
            WP_2 = [2.0,0.0]
            WP_3 = [0.0,2.0]
            WP_4 = [2.0,2.0]
            
        IP = np.float32([IP_1, IP_2, IP_3, IP_4])
        WP = np.float32([WP_1, WP_2,WP_3,WP_4])
        return IP, WP

    def ComputeVanishingPointPT(self, IP):

        Up_horLine = np.array([IP[0],IP[1]])
        Bottom_horLine = np.array([IP[2],IP[3]])
        Left_vertLine = np.array([IP[0],IP[2]])
        Right_vertLine = np.array([IP[1],IP[3]])

        VP1 = self.line_intersection(Right_vertLine, Left_vertLine)
        VP2 = self.line_intersection(Up_horLine, Bottom_horLine)

        VP1 = np.array(VP1)/self.factor
        VP2 = np.array(VP2)/self.factor
        return VP1, VP2

    def computeCameraEye_PT(self, image_points, world_points):
        focal_length = 25
        pixel_size = 0.00345
        image_resolution = [2464, 2056]
        image2world = cv2.getPerspectiveTransform(image_points, world_points)

        intrinsic = [[focal_length / pixel_size, 0, image_resolution[0] / 2],
                    [0, focal_length / pixel_size, image_resolution[1] / 2],
                    [0, 0, 1]]

        world_points_3D = []
        image_points_2D = []
        for elem in world_points : 
            world_points_3D.append([elem[0], elem[1], 0])

        for elem in image_points : 
            image_points_2D.append([elem[0], elem[1], 0])
        
        retVal, rvec, tvec = cv2.solvePnP(np.array(world_points_3D), np.array(image_points), np.array(intrinsic), None)
        if retVal :
            rotation_matrix = cv2.Rodrigues(np.array(rvec))[0]
            cameraEye = - np.transpose(rotation_matrix) @ tvec 

        return image2world, cameraEye

class VanishingPointCalibration():
    def __init__(self):
        self.imgsz_init = np.array([2464,2056])
        self.imgsz = np.array([320,320])
        self.factor = self.imgsz_init/self.imgsz
    def get_KP(self, cam):
        if cam == "staging":
            P = [[0.161577, -0.986568, -0.023923, -0.000000], [-0.174364, -0.052394, 0.983283, 0.000380], [0.971237, 0.154664, 0.180509, -25483.884766]] #extrinsic

            K = [[8314.529297, 0.000000, 1232.000000], [0.000000, 8314.529297, 1028.000122], [0.000000, 0.000000, 1.000000]] #intrinsic
        if cam == "haunter":
            P = [[-0.148152, 0.987029, 0.061890, -0.003200], [-0.160669, -0.085777, 0.983267, -0.019003], [0.975544, 0.135700, 0.171340, -26847.136719]] #extrinsic

            K = [[9760.676758, 0.000000, 1231.998779], [0.000000, 9760.676758, 1027.993042], [0.000000, 0.000000, 1.000000]] #intrinsic
        if cam == "caterpie":
            P = [[0.013557, 0.999568, -0.025042, -0.000755], [-0.789422, 0.025988, 0.613220, 0.000755], [0.613560, 0.011518, 0.789521, -12665.919922]] #extrinsic

            K = [[1368.062744, 0.000000, 1232.000000], [0.000000, 1368.062744, 1028.000122], [0.000000, 0.000000, 1.000000]] #intrinsic

        return K, P

    def get_VP(self, cam):
        #Vanishing point for one type of images (Camera nvidia@10.2.40.68) staging
        #Unit : Pixels 
        if cam == "staging": 
            vp1 = np.array([2536.74,-413.156])/self.factor 
            vp2 = np.array([-17456.0,273.684])/self.factor
        elif cam == "hypno":
            # #Vanishing point for one type of images (Camera nvidia@10.2.40.97) hypno
            vp1 = np.array([-802.421,-973.685])/self.factor 
            vp2 = np.array([33457.8,-2238.63])/self.factor
        elif cam == "haunter":
            # #Vanishing point for one type of images (Camera nvidia@10.2.40.93) haunter
            vp1 = np.array([-196.768,-512.267])/self.factor 
            vp2 = np.array([170434.,-16098.9])/self.factor
        elif cam == "highway":
            vp1 = np.array([1136.0663366336635, -771.6970297029704])/self.factor 
            vp2 = np.array([-19000,1028])/self.factor #NEEDS TO FIND THE SECOND ONE
        elif cam == "caterpie":
            vp2 = np.array([1262.23,-732.184])/self.factor 
            vp1 = np.array([119955.,1200.76])/self.factor 

            # vp1 = np.array([1136.0663366336635, -771.6970297029704])/self.factor 
            # vp2 = np.array([-19000,1028])/self.factor #NEEDS TO FIND THE SECOND ONE
            # vp3 (1188.61,2090.57)
        else:
            print("You have to select which camera you are using : Staging, haunter or Hypno")
            exit(0)
        return vp1, vp2
    
    def updateIntrinsicExtrinsic(self, K, P):
        extrinsic = np.array(P)
        intrinsic = np.array(K)
        world2image = np.dot(intrinsic,extrinsic)

        M = np.zeros((3,3))
        for i in range(0,3):
            M[i,0] = world2image[i,0]
            M[i,1] = world2image[i,1]
            M[i,2] = world2image[i,3]

        image2world = np.linalg.inv(M)
        R_inv = np.linalg.inv(world2image[0:3,0:3])
        t = world2image[0:3,3:4]
        cameraEye = -1*np.dot(R_inv, t)

        return image2world, cameraEye

class BOX_3D(PerspectiveTransformCalibration, VanishingPointCalibration):
    def __init__(self, ):
        super().__init__()
        self.imgsz_init = np.array([2464,2056])
        self.imgsz = np.array([320,320])
        self.factor = self.imgsz_init/self.imgsz
    def PreProcess(self, img_path, real_image_path):
        if len(img_path) == 1:
                img_path = img_path[0]
        if len(real_image_path) == 1:
            real_image_path =real_image_path[0]
        image = cv2.imread(img_path)
        real_image  = cv2.imread(real_image_path)
        return image, real_image

    def projectImageToWorld2D(self, pixel_coord, image2world):
        cam_coord = np.zeros((3,1))
        cam_coord[0] = pixel_coord[0]
        cam_coord[1] = pixel_coord[1]
        cam_coord[2] = 1
        world_coord = np.dot(image2world, cam_coord)
        point2f = np.array([world_coord[0]/world_coord[2], world_coord[1]/world_coord[2]])
        return point2f

    def  findMostPoint(self,true_value):
        current_P_right = [-1e10, -1e10]
        current_P_left = [+1e10, +1e10]

        for loc in true_value:
            if loc[0] > current_P_right[0]:
                current_P_right = loc
            if loc[0] < current_P_left[0]:
                current_P_left = loc
        return np.array(current_P_right), np.array(current_P_left)

    def getVerticalLines(self, RightPoint, LeftPoint):
        LeftLine = np.array([LeftPoint,np.array([LeftPoint[0], 320])])
        RightLine = np.array([RightPoint,np.array([RightPoint[0], 320])])
        return LeftLine, RightLine

    def ComputeIntersection(self, VP_info):
     
        A = self.line_intersection(VP_info['vp1_leftTangentLine'], VP_info['vp0_rightTangentLine'])
        B = self.line_intersection(VP_info['vp1_leftTangentLine'], VP_info['LeftHorizontalLine'])
        C = self.line_intersection(VP_info['vp0_rightTangentLine'], VP_info['RightHorizontalLine'])
        D = self.line_intersection(VP_info['vp1_rightTangentLine'], VP_info['RightHorizontalLine'])
        F = self.line_intersection(VP_info['vp0_leftTangentLine'], VP_info['LeftHorizontalLine'])
        
        
        LineD_vp0 = np.array([VP_info['vp0_coordinate'], D])
        LineF_vp1 = np.array([VP_info['vp1_coordinate'], F])
        LineD_vp1 = np.array([VP_info['vp1_coordinate'], D])
        LineF_vp0 = np.array([VP_info['vp0_coordinate'], F])
        LineC_vp1 = np.array([VP_info['vp1_coordinate'], C])
        LineB_vp0 = np.array([VP_info['vp0_coordinate'], B])
        LineC_vp0 = np.array([VP_info['vp0_coordinate'], C])
        LineB_vp1 = np.array([VP_info['vp1_coordinate'], B])

        Line_AW = np.array([A, [A[0],320]])
        E = self.line_intersection(LineD_vp0, LineF_vp1)
        E_D = self.line_intersection(LineD_vp0, Line_AW)
        E_F = self.line_intersection(LineF_vp1, Line_AW)
        G = self.line_intersection(LineD_vp1, LineF_vp0)
        Line_GW = np.array([G, [G[0],320]])
        H_C = self.line_intersection(LineC_vp1, Line_GW)
        H_B = self.line_intersection(LineB_vp0, Line_GW)
        H = self.line_intersection(LineB_vp0, LineC_vp1)

        return A, B, C, D, F, E_F, G, H

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def CLTVP(self, vp1, vp2, mask, image_test):
        VP_info = {}
        VP = [vp1, vp2]
        true_value = []  
        for elem in mask:
            for loc in elem:
                true_value.append(loc[0])
        rightMost, LeftMost = self.findMostPoint(true_value)
        LeftHorizontalLine, RightHorizontalLine = self.getVerticalLines(rightMost, LeftMost)
        VP_info["LeftHorizontalLine"] = LeftHorizontalLine
        VP_info["RightHorizontalLine"] = RightHorizontalLine
        for z in range(len(VP)):
            minAngle, maxAngle, leftTangentLine, rightTangentLine = 1e10, -1e10, np.array([[0,0],[0,0]]), np.array([[0,0],[0,0]])  
            for loc in true_value:
                currentLine = [np.array(loc), np.array([int(VP[z][0]), int(VP[z][1])])]
                angle = math.atan2((loc[0]-int(VP[z][0])),(loc[1]-int(VP[z][1])))
                if angle < 0 and rightMost[0] < int(VP[z][0]):
                        angle = angle + 2*math.pi
                if angle < minAngle :
                    minAngle = angle
                    leftTangentLine = currentLine
                if angle > maxAngle :
                    maxAngle = angle
                    rightTangentLine = currentLine
            VP_info["vp{0}_coordinate".format(z)] = np.array([int(VP[z][0]), int(VP[z][1])])
            VP_info["vp{0}_leftTangentLine".format(z)] = leftTangentLine
            VP_info["vp{0}_rightTangentLine".format(z)] = rightTangentLine
        A, B, C, D, F, E, G, H = self.ComputeIntersection(VP_info)
        return  A, B, C, D, F, E, G, H, image_test 

    def heightComputation(self, Points_image, image2world, cameraEye):
        Points_world, Distance_Eye, list_dist_points, list_height,  = [], [], [], []
        for elem in Points_image:
            Points_world.append(self.projectImageToWorld2D(elem, image2world))
        for elem in Points_world:
            D_Cam = math.hypot(elem[0] - cameraEye[0], elem[1] - cameraEye[1])
            Distance_Eye.append(D_Cam)
        list_dist_points = []
        for i, elem in enumerate(Points_image):
            norm = 1e10
            for j, elem_2 in enumerate(Points_image):
                temp = elem[0] - elem_2[0]
                if abs(temp) < norm and i != j:
                    norm = abs(temp)
                    index = [i,j]
            if Points_image[index[0]][1] < Points_image[index[1]][1]:
                dist_i_j = abs(Distance_Eye[index[0]] - Distance_Eye[index[1]])
                list_dist_points.append([dist_i_j, Distance_Eye[index[0]], Distance_Eye[index[1]]])
                height_i_j =  abs((cameraEye[2][0]/Distance_Eye[index[0]]) * dist_i_j)
                list_height.append(height_i_j)
        return Points_world, Distance_Eye, list_dist_points, list_height

    def widthComputation(self, Points_image, image2world):
        Points_world, Distance_Eye, list_dist_points, list_height,  = [], [], [], []
        for elem in Points_image:
            Points_world.append(self.projectImageToWorld2D(elem, image2world))
        for i, elem in enumerate(Points_image):
            norm = 1e10
            for j, elem_2 in enumerate(Points_image):
                print()

    def CannyEdgeDetector(self, mask):
        mask_C = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ones  = sum(map(sum,mask_C))
        if ones/255 > 1e4:
             kernel = np.ones((9,9),np.uint8)
        if ones/255 <= 1e4:
             kernel = np.ones((5,5),np.uint8)
        mask_C = cv2.morphologyEx(mask_C, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("results/car_images/images_.jpg", mask_C)
        mask = cv2.Canny(mask_C,254,255)
        cv2.imwrite("results/car_images/images_mask.jpg", mask)
        mask, hierrachy  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return mask
    def Box3D(self, image, real_image, mask, save_image, cam, CarDimension, method):
        
        # real_image = cv2.flip(real_image, 1)
        # mask = cv2.flip(mask, 1)
        image_test = real_image
        
        mask  = self.CannyEdgeDetector(mask)
        if method == "VP_calibration" :
            vp1, vp2 = self.get_VP(cam)
        elif method  == "PT_calibration" :  
            IP, WP = self.getIP_WP(cam) 
            vp1, vp2 =self. ComputeVanishingPointPT(IP)
        else:
            print("You have to choose a calibration method between Vanishing Point Calibration (flag : VP_calibration) or Perspective Transform Calibration (flag : PT_calibration)!")
            exit(0)
        A, B, C, D, F, E, G, H, image_test = self.CLTVP(vp1, vp2, mask, image_test)
        imgsz = np.array([320,256])
        Points_image =  list((A, B, C, D, E, F, G, H))
        factor = self.imgsz_init/imgsz
        A, B, C, D, F, E, G, H = (((A*factor)), ((B*factor)), (C*factor), ((D*factor)), ((F*factor)), ((E*factor)), ((G*factor)), ((H*factor)))
        A[1], B[1], C[1], D[1], F[1], E[1], G[1], H[1] = ((A[1])-30*(factor[1])), ((B[1])-30*(factor[1])), ((C[1])-(30*factor[1])), ((D[1])-30*(factor[1])), ((F[1])-30*(factor[1])), ((E[1])-30*(factor[1])), ((G[1])-30*(factor[1])), ((H[1])-30*(factor[1]))
        for i, elem in enumerate(Points_image):
            elem = elem*factor
            elem[1] = ((elem[1])-30*(factor[1]))
            Points_image[i] = elem

        if CarDimension: 
            if method == "VP_calibration":
                K, P = self.get_KP(cam)
                image2world, cameraEye = self.updateIntrinsicExtrinsic(K, P)
            if method == "PT_calibration":
                image2world, cameraEye = self.computeCameraEye_PT(IP, WP)
                if True :
                    cameraEye[2] = cameraEye[2] - 0.8
            point2f_A = self.projectImageToWorld2D(A, image2world)
            point2f_C = self.projectImageToWorld2D(C, image2world)
            point2f_B = self.projectImageToWorld2D(B, image2world)
            point2f_D = self.projectImageToWorld2D(D, image2world)
            point2f_E = self.projectImageToWorld2D(E, image2world)
            point2f_F = self.projectImageToWorld2D(F, image2world)
            point2f_G = self.projectImageToWorld2D(G, image2world)
            point2f_H = self.projectImageToWorld2D(H, image2world)

            Points_world, Distance_Eye, list_dist_points, list_height = self.heightComputation(Points_image, image2world, cameraEye)

            length_1 = math.hypot(point2f_A[0] - point2f_C[0], point2f_A[1] - point2f_C[1])
            length_2 = math.hypot(point2f_B[0] - point2f_H[0], point2f_B[1] - point2f_H[1])
            length_3 = math.hypot(point2f_E[0] - point2f_D[0], point2f_E[1] - point2f_D[1])
            length_4 = math.hypot(point2f_F[0] - point2f_G[0], point2f_F[1] - point2f_G[1])
            width_1 = math.hypot(point2f_A[0] - point2f_B[0], point2f_A[1] - point2f_B[1])
            width_2 = math.hypot(point2f_C[0] - point2f_H[0], point2f_C[1] - point2f_H[1])
            width_3 = math.hypot(point2f_F[0] - point2f_E[0], point2f_F[1] - point2f_E[1])
            width_4 = math.hypot(point2f_G[0] - point2f_D[0], point2f_G[1] - point2f_D[1])
            print("Perspective Point Calibration : Length of the car : ", length_1, " width of the car : ", width_1," height of the car : ",  list_height[0])
            print("Perspective Point Calibration : Length of the car : ", length_2, " width of the car : ", width_2," height of the car : ",  list_height[1])
            print("Perspective Point Calibration : Length of the car : ", length_3, " width of the car : ", width_3," height of the car : ",  list_height[2])
            # print("Perspective Point Calibration : Length of the car : ", length_4, " width of the car : ", width_4," height of the car : ",  list_height[3])
            
        blue = [255,0,0]
        green = [0,255,0]
        red = [0,0,255]
        if save_image:
            cv2.line(image_test,tuple([int(E[0]), int(E[1])]),tuple([int(A[0]), int(A[1])]), red, 3)
            cv2.line(image_test,tuple([int(C[0]), int(C[1])]),tuple([int(A[0]), int(A[1])]), blue, 3)
            cv2.line(image_test,tuple([int(B[0]), int(B[1])]),tuple([int(A[0]), int(A[1])]), green, 3)
            cv2.line(image_test,tuple([int(E[0]), int(E[1])]),tuple([int(D[0]), int(D[1])]), blue, 3)
            cv2.line(image_test,tuple([int(E[0]), int(E[1])]),tuple([int(F[0]), int(F[1])]), green, 3)
            cv2.line(image_test,tuple([int(D[0]), int(D[1])]),tuple([int(C[0]), int(C[1])]), red, 3)
            cv2.line(image_test,tuple([int(F[0]), int(F[1])]),tuple([int(B[0]), int(B[1])]), red, 3)
            cv2.line(image_test,tuple([int(D[0]), int(D[1])]),tuple([int(G[0]), int(G[1])]), green, 3)
            cv2.line(image_test,tuple([int(F[0]), int(F[1])]),tuple([int(G[0]), int(G[1])]), blue, 3)
            cv2.line(image_test,np.array([int(H[0]), int(H[1])]),np.array([int(B[0]), int(B[1])]), blue, 2)
            cv2.line(image_test,np.array([int(H[0]), int(H[1])]),np.array([int(G[0]), int(G[1])]), red, 2)
            cv2.line(image_test,np.array([int(H[0]), int(H[1])]),np.array([int(C[0]), int(C[1])]), green, 2)
            
            
            cv2.putText(image_test,str((int(length_1/10)/100)), (int(F[0])-30,int(F[1])-90), cv2.FONT_HERSHEY_SIMPLEX, 2, 255,3)
            cv2.putText(image_test,str((int(list_height[0]/10)/100)), (int(B[0])-150,int(B[1])-180), cv2.FONT_HERSHEY_SIMPLEX, 2, red,3)
            cv2.putText(image_test,str((int(width_1/10)/100)), (int(B[0])+180,int(B[1])+70), cv2.FONT_HERSHEY_SIMPLEX, 2, green,3)
        return image_test
            

def get_Parser():
    parser = argparse.ArgumentParser(
            description="Creation of bounding boxes around segmented object")
    parser.add_argument(
            "--input",
            default="results/car_images/320_trt_cv2img_3.jpg",
            type=str,
            nargs="+",
            help="Directory to input images",
            )
    parser.add_argument(
            "--real_input",
            default="inference/vp.jpg.jpg",
            type=str,
            nargs="+",
            help="Directory to input images",
            )
    parser.add_argument(
            "--mask_path",
            default="results/car_images/mask/320_trt_cv2img_mask_3.jpg",
            type=str,
            nargs="+",
            help="Directory to mask of input images",
            )
    parser.add_argument(
            "--cam",
            default="None",
            type=str,
            help="staging, haunter or hypno",
            )
    parser.add_argument(
            "--contours",
            action='store_true',
            help="Directory to mask of input images",
            )
    parser.add_argument(
            "--cheat",
            action='store_true',
            help="Directory to mask of input images",
            )
    parser.add_argument(
            "--method",
            default="None",
            type=str,
            help="VP_calibration, PT_calibration",
            )
    parser.add_argument(
            "--CarDimension",
            action='store_true',
            )
    parser.add_argument(
            "--save",
            action='store_true',
            )
    return parser

if __name__ == '__main__':
    args = get_Parser().parse_args()
    img_path = args.input
    mask_path = args.mask_path
    real_image_path = args.real_input
    BOX = BOX_3D()
    VanishingPointCalibration()
    image, real_image = BOX.PreProcess(img_path, real_image_path)
    
    mask  = cv2.imread(mask_path[0])
    real_image = BOX.Box3D(image, real_image, mask, args.save, args.cam, args.CarDimension, args.method)
