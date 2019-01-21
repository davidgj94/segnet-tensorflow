import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import math
from collections import Counter
import vis
from matplotlib.patches import Circle

palette = np.array([[255,255,255],[0, 0, 255],[0, 255, 0]])


def general_form(rho, theta):

    a = math.cos(theta)
    b = math.sin(theta)
    c = -rho

    return (a,b,c)

def normal_form(a,b,c):

    theta = math.atan(b/a)
    rho = -c
    return (rho, theta)


def dist_parallel_lines(line1, line2):

    rho1, theta1 = line1
    rho2, theta2 = line2

    a, b, _ = general_form(rho1, theta1)

    dist = abs(rho1-rho2) / math.sqrt(a ** 2 + b ** 2)

    return dist

def get_parallel_line(line, d):

    rho1, theta1 = line
    a, b, _ = general_form(rho1, theta1)
    rho2 = rho1 - d * math.sqrt(a ** 2 + b ** 2)

    print rho2

    return (rho2, theta1)


def find_intersect(line1_coeffs, line2_coeffs):

    A1, B1, C1 = line1_coeffs
    A2, B2, C2 = line2_coeffs

    denom = (A1 * B2 - B1 * A2)

    if abs(denom) > 1e-10:
        x = (B1 * C2 - C1 * B2) / denom
        y = (C1 * A2 - A1 * C2) / denom
    else:
        return None

    return (x, y)

def get_line_coeffs(point, orientation):
    x, y = point
    A = math.cos(orientation)
    B = math.sin(orientation)
    C = -(A * x + B * y)
    return (A, B, C)

def check_intersect(point, sz):

    H, W = sz

    if point:
        exists_intersect = (0.0 <= point[0] <= float(W)) and (0.0 <= point[1] <= float(H))
    else:
        exists_intersect = False

    return exists_intersect

def find_intesect_borders(line_coeffs, sz):

    H, W = sz

    upper_border_coeffs = (0.0, 1.0, 0.0)
    lower_border_coeffs = (0.0, 1.0, -float(H))
    left_border_coeffs = (1.0, 0.0, 0.0)
    right_border_coeffs = (1.0, 0.0, -float(W))

    upper_border_intersect = find_intersect(line_coeffs, upper_border_coeffs)
    lower_border_intersect = find_intersect(line_coeffs, lower_border_coeffs)
    left_border_intersect = find_intersect(line_coeffs, left_border_coeffs)
    right_border_intersect = find_intersect(line_coeffs, right_border_coeffs)

    intersect_points = []
    if check_intersect(upper_border_intersect, sz):
        intersect_points.append(upper_border_intersect)
    if check_intersect(lower_border_intersect, sz):
        intersect_points.append(lower_border_intersect)
    if check_intersect(left_border_intersect, sz):
        intersect_points.append(left_border_intersect)
    if check_intersect(right_border_intersect, sz):
        intersect_points.append(right_border_intersect)

    return intersect_points

filter = True

def fix_lines(lines_v, sz):

    checked_lines = [False] * len(lines_v)
    new_lines_v = []

    for i in range(len(lines_v)):

        if checked_lines[i]:
            continue

        print lines_v[i]
        print math.degrees(lines_v[i][1])
        print
        line_coeffs_i = general_form(*lines_v[i])
        checked_lines[i] = True

        intersection = False
        for j in range(len(lines_v)):

            if (i != j) and not checked_lines[j]:

                line_coeffs_j = general_form(*lines_v[j])
                intersection_point = find_intersect(line_coeffs_i, line_coeffs_j)

                if check_intersect(intersection_point, sz):

                    print "inter"
                    print lines_v[j]
                    print math.degrees(lines_v[j][1])
                    print

                    # print intersection_point

                    checked_lines[j] = True
                    intersection = True
                    theta1 = math.degrees(lines_v[i][1])
                    theta2 = math.degrees(lines_v[j][1])
                        
                    # if (170.0 <= theta1 <= 180.0):
                    #     theta1 = 180 - theta1
                    # if (170.0 <= theta2 <= 180.0):
                    #     theta2 = 180 - theta2

                    new_theta = (theta1 + theta2) / 2
                    # print (theta1, theta2)
                    # print (lines_v[i][0], lines_v[j][0])

                    if new_theta < 0:
                        new_theta = new_theta + 180

                    # print new_theta

                    new_line_coeffs = get_line_coeffs(intersection_point, math.radians(new_theta))
                    # print normal_form(*new_line_coeffs)
                    new_lines_v.append(normal_form(*new_line_coeffs))

        if not intersection:
            new_lines_v.append(lines_v[i])

    return new_lines_v


#file_path = '009-APR-20-2-90/masks/frame_60.png'
# file_path_mask = '009-APR-20-2-90/masks/frame_116.png'
# file_path_img = '009-APR-20-2-90/imgs/frame_116.png'
file_path_mask = '009-APR-20-2-90/masks/frame_2.png'
file_path_img = '009-APR-20-2-90/imgs/frame_2.png'
edges_ = cv2.imread(file_path_mask)
image = cv2.imread(file_path_img)
edges = edges_[...,0]
#pdb.set_trace()
# plt.figure()
# plt.imshow(edges)
kernel = np.ones((5,5), np.uint8)

#find all your connected components (white blobs in your image)
edges_aux = np.zeros(edges.shape, dtype=np.uint8)
edges_aux[edges == 1] = 255

# edges_aux = cv2.dilate(edges_aux, kernel, iterations=2)

# plt.figure()
# plt.imshow(edges_aux == 255)

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(edges_aux, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1
sizes_sorted_indices = np.argsort(sizes)
blob = np.zeros(output.shape, dtype=np.uint8)
blob[output == (sizes_sorted_indices[-1] + 1)] = 255

if sizes[sizes_sorted_indices[-2]] >= (sizes[sizes_sorted_indices[-1]] / 2):
    blob[output == (sizes_sorted_indices[-2] + 1)] = 255

# plt.figure()
# plt.imshow(blob == 255)

blob = cv2.erode(blob, kernel, iterations=2)

# plt.figure()
# plt.imshow(blob == 255)
#plt.show()

# plt.figure()
# plt.imshow(blob == 255)
# plt.show()

#lines = cv2.HoughLines(blob,1,np.pi/180,500)
lines = []
for line in cv2.HoughLines(blob,1,np.pi/180, 250):
    rho, theta = line[0]
    if rho < 0:
        print rho
        print math.degrees(theta)
        new_rho = -1 * rho
        new_theta = theta + np.pi
        lines.append([[new_rho, new_theta]])
    else:
        lines.append(line)

pdb.set_trace()
lines = np.array(lines)

if not lines.any():
    print('No lines were found')
    exit()

if filter:
    rho_threshold = 50
    theta_threshold = 0.5

    # how many lines are similar to a given one
    similar_lines = {i : [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i,theta_i = lines[i][0]
            rho_j,theta_j = lines[j][0]
            diff_rho = abs(rho_i - rho_j)
            diff_theta = abs(theta_i - theta_j)
            if diff_rho < rho_threshold and min(diff_theta, 2*np.pi - diff_theta) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x : len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines)*[True]
    for i in range(len(lines) - 1):
        if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
            continue

        for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
            if not line_flags[indices[j]]: # and only if we have not disregarded them already
                continue

            rho_i,theta_i = lines[indices[i]][0]
            rho_j,theta_j = lines[indices[j]][0]
            diff_rho = abs(rho_i - rho_j)
            diff_theta = abs(theta_i - theta_j)
            if diff_rho < rho_threshold and min(diff_theta, 2*np.pi - diff_theta) < theta_threshold:
                line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

print('number of Hough lines:', len(lines))

filtered_lines = []

if filter:
    for i in range(len(lines)): # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])

    print('Number of filtered lines:', len(filtered_lines))
else:
    filtered_lines = lines

line_aux = np.zeros(edges_.shape, dtype=np.uint8)
line_aux2 = np.zeros(edges_.shape, dtype=np.uint8)
lines_h = []
thetas_h = []
lines_v = []
thetas_v = []
for line in filtered_lines:

    rho,theta = line[0]

    theta_degrees = math.degrees(theta)
    theta_degrees = min(theta_degrees, 360.0 - theta_degrees)
    print theta_degrees
    if (80.0 <= theta_degrees <= 100.0):
        lines_h.append((rho, theta))
        thetas_h.append(theta_degrees)

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)

    elif (0.0 <= theta_degrees <= 10.0):
        lines_v.append((rho, theta))
        thetas_v.append(theta_degrees)

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(line_aux2,(x1,y1),(x2,y2),(0,0,255),2)

grid2 = np.zeros(edges.shape, dtype=np.uint8)
grid2[line_aux2[...,-1] == 255] = 1

vis_img2 = vis.vis_seg(np.squeeze(image[...,::-1]), grid2, palette)
plt.figure()
plt.imshow(vis_img2)
plt.show()

pdb.set_trace()


new_lines_h = fix_lines(lines_h, edges.shape)
all_lines_mask = np.zeros(edges_.shape, dtype=np.uint8)
for line in new_lines_h:

    rho,theta = line

    print find_intesect_borders(general_form(*line), edges.shape)

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    print (x1,y1)
    print (x2,y2)
    cv2.line(all_lines_mask,(x1,y1),(x2,y2),(0,0,255),2)

grid_all = np.zeros(edges.shape, dtype=np.uint8)
grid_all[all_lines_mask[...,-1] == 255] = 1

vis_img_all = vis.vis_seg(np.squeeze(image[...,::-1]), grid_all, palette)
plt.figure()
plt.imshow(vis_img_all)
plt.show()



lines_h_sorted = sorted(lines_h, key= lambda x: x[0])
# line_aux_ = np.zeros(edges_.shape, dtype=np.uint8)
# for line in lines_h_sorted:

#     rho, theta = line
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv2.line(line_aux_,(x1,y1),(x2,y2),(0,0,255),2)

#     grid2 = np.zeros(edges.shape, dtype=np.uint8)
#     grid2[line_aux_[...,-1] == 255] = 1
#     vis_img2 = vis.vis_seg(np.squeeze(image[...,::-1]), grid2, palette)
#     plt.figure()
#     plt.imshow(vis_img2)
#     plt.show()



lines_v_sorted = sorted(lines_v, key= lambda x: x[0])



for i in range(len(lines_v_sorted)-1):

    line1 = lines_v_sorted[i]
    rho, theta = line1

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)
    grid2 = np.zeros(edges.shape, dtype=np.uint8)
    grid2[line_aux[...,-1] == 255] = 1
    vis_img2 = vis.vis_seg(np.squeeze(image[...,::-1]), grid2, palette)
    plt.figure()
    plt.imshow(vis_img2)
    plt.show()

    line2 = lines_v_sorted[i+1]
    dist = dist_parallel_lines(line1, line2)
    num_inter_lines = int(round(dist / 150.0)) - 1

    print dist
    print num_inter_lines
    D = dist / (num_inter_lines + 1)
    print D

    if num_inter_lines > 0:

        num_parallel_1 = num_inter_lines / 2 + num_inter_lines % 2
        num_parallel_2 = num_inter_lines / 2 

        # if line1[0] > 0:
        #     d = -D
        # else:
        #     d = D

        d = -D

        for _ in range(num_parallel_1):
            rho, theta = get_parallel_line(line1, d)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)
            grid2 = np.zeros(edges.shape, dtype=np.uint8)
            grid2[line_aux[...,-1] == 255] = 1
            vis_img2 = vis.vis_seg(np.squeeze(image[...,::-1]), grid2, palette)
            plt.figure()
            plt.imshow(vis_img2)
            plt.show()
            d -= D
            # if line1[0] > 0:
            #     d -= D
            # else:
            #     d += D

        # if line2[0] > 0:
        #     d = D
        # else:
        #     d = -D
        d = D

        for _ in range(num_parallel_2):
            rho, theta = get_parallel_line(line2, d)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)
            grid2 = np.zeros(edges.shape, dtype=np.uint8)
            grid2[line_aux[...,-1] == 255] = 1
            vis_img2 = vis.vis_seg(np.squeeze(image[...,::-1]), grid2, palette)
            plt.figure()
            plt.imshow(vis_img2)
            plt.show()
            d += D
            # if line2[0] > 0:
            #     d += D
            # else:
            #     d -= D



   


rho, theta = lines_v_sorted[-1]
a = np.cos(theta)
b = np.sin(theta)
x0 = a*rho
y0 = b*rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*(a))
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*(a))
cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)

    
# if lines_v_sorted[0] > 0:
#     d = -150
# else:
#     d = +150

d = 150

rho, theta = get_parallel_line(lines_v_sorted[0], d)

if find_intesect_borders(general_form(rho, theta), edges.shape):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)


# if lines_v_sorted[-1] > 0:
#     d = -150
# else:
#     d = +150

d = -150

rho, theta = get_parallel_line(lines_v_sorted[-1], d)

if find_intesect_borders(general_form(rho, theta), edges.shape):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)



grid2 = np.zeros(edges.shape, dtype=np.uint8)
grid2[line_aux[...,-1] == 255] = 1
vis_img2 = vis.vis_seg(np.squeeze(image[...,::-1]), grid2, palette)
plt.figure()
plt.imshow(vis_img2)
plt.show()











# def fix_lines(lines_v, sz):

#     checked_lines = [False] * len(lines_v)
#     new_lines_v = []

#     for i in range(len(lines_v)):

#         if checked_lines[i]:
#             continue

#         line_coeffs_i = general_form(*lines_v[i])
#         checked_lines[i] = True

#         intersection = False
#         for j in range(len(lines_v)):

#             if (i != j) and not checked_lines[j]:

#                 line_coeffs_j = general_form(*lines_v[j])
#                 intersection_point = find_intersect(line_coeffs_i, line_coeffs_j)

#                 if check_intersect(intersection_point, sz):

#                     print intersection_point

#                     checked_lines[j] = True
#                     intersection = True
#                     theta1 = math.degrees(lines_v[i][1])
#                     theta2 = math.degrees(lines_v[j][1])
                        
#                     if (170.0 <= theta1 <= 180.0):
#                         theta1 = 180 - theta1
#                     if (170.0 <= theta2 <= 180.0):
#                         theta2 = 180 - theta2

#                     new_theta = (theta1 + theta2) / 2
#                     print (theta1, theta2)
#                     print (lines_v[i][0], lines_v[j][0])

#                     if new_theta < 0:
#                         new_theta = new_theta + 180

#                     print new_theta

#                     new_line_coeffs = get_line_coeffs(intersection_point, math.radians(new_theta))
#                     print normal_form(*new_line_coeffs)
#                     new_lines_v.append(normal_form(*new_line_coeffs))

#         if not intersection:
#             new_lines_v.append(lines_v[i])

#     return new_lines_v

# new_lines_v = fix_lines(lines_v, edges.shape)
# new_lines_h = fix_lines(lines_h, edges.shape)
# #pdb.set_trace()
# all_lines = new_lines_v + new_lines_h
# all_lines_mask = np.zeros(edges_.shape, dtype=np.uint8)
# for line in all_lines:

#     rho,theta = line

#     print find_intesect_borders(general_form(*line), edges.shape)

#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     print (x1,y1)
#     print (x2,y2)
#     cv2.line(all_lines_mask,(x1,y1),(x2,y2),(0,0,255),2)

# grid_all = np.zeros(edges.shape, dtype=np.uint8)
# grid_all[all_lines_mask[...,-1] == 255] = 1

# vis_img_all = vis.vis_seg(np.squeeze(image[...,::-1]), grid_all, palette)
# plt.figure()
# plt.imshow(vis_img_all)
# plt.show()



# grid2 = np.zeros(edges.shape, dtype=np.uint8)
# grid2[line_aux2[...,-1] == 255] = 1

# vis_img2 = vis.vis_seg(np.squeeze(image[...,::-1]), grid2, palette)
# plt.figure()
# plt.imshow(vis_img2)



# theta_h = Counter(thetas_h).most_common()[0][0]
# lines_h_final = [line for line, theta_line in zip(lines_h, thetas_h) if abs(theta_line - theta_h) < 1e-6]

# if theta_h >= 90.0:
#     lines_v_final = [line for line, theta_line in zip(lines_v, thetas_v) if abs(theta_line - (theta_h - 90.0)) < 2.0]
#     diff_v_final = [abs(theta_line - (theta_h - 90.0)) for line, theta_line in zip(lines_v, thetas_v) if abs(theta_line - (theta_h - 90.0)) < 2.0]
# else:
#     lines_v_final = [line for line, theta_line in zip(lines_v, thetas_v) if abs(theta_line - (180 + (theta_h - 90.0))) < 2.0]
#     diff_v_final = [abs(theta_line - (180 + (theta_h - 90.0))) for line, theta_line in zip(lines_v, thetas_v) if abs(theta_line - (180 + (theta_h - 90.0))) < 2.0]

# anchor_h = lines_h_final[0]
# min_diff_index = np.argmin(diff_v_final)
# anchor_v = lines_v_final[min_diff_index]


# intersect_point = find_intersect(general_form(*anchor_h), general_form(*anchor_v))
# circ1 = Circle(intersect_point,3)
# fig1,ax1 = plt.subplots(1)
# ax1.imshow(image)
# ax1.add_patch(circ1)
# plt.show()

# if theta_h >= 90.0:
#     anchor_coeffs = get_line_coeffs(intersect_point, math.radians(theta_h - 90.0))
# else:
#     anchor_coeffs = get_line_coeffs(intersect_point, math.radians(180 + (theta_h - 90.0)))

# anchor_v = normal_form(*anchor_coeffs)

# rho, theta = anchor_v
# a = np.cos(theta)
# b = np.sin(theta)
# x0 = a*rho
# y0 = b*rho
# x1 = int(x0 + 1000*(-b))
# y1 = int(y0 + 1000*(a))
# x2 = int(x0 - 1000*(-b))
# y2 = int(y0 - 1000*(a))
# cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)

# intersection = True
# d = 150.0
# while intersection:

#     rho, theta = get_parallel_line(anchor_v, d)

#     if find_intesect_borders(general_form(rho, theta), edges.shape):
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#         cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)
#         d += 150.0
#     else:
#         intersection = False

# # plt.figure()
# # plt.imshow(line_aux)

# intersection = True
# d = -150.0
# while intersection:

#     rho, theta = get_parallel_line(anchor_v, d)

#     if find_intesect_borders(general_form(rho, theta), edges.shape):
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#         cv2.line(line_aux,(x1,y1),(x2,y2),(0,0,255),2)
#         d -= 150.0
#     else:
#         intersection = False
    

# grid = np.zeros(edges.shape, dtype=np.uint8)
# grid[line_aux[...,-1] == 255] = 1

# vis_img = vis.vis_seg(np.squeeze(image[...,::-1]), grid, palette)

# plt.figure()
# plt.imshow(vis_img)
# plt.show()
