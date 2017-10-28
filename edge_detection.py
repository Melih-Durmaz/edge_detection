import os
import numpy as np

# Applies min-max normalization to the given parameter to interval of [0,255]
# After normalization, the value si converted to integer and returned.


class EdgeDetection:

    def detect_edges(self,img_name):
        file = FileIO()

        img_matrix, size = file.read_pgm_to_matrix(img_name)
        self.ed_filter(img_matrix, size, img_name, filter_type='prewitt')
        self.ed_filter(img_matrix, size, img_name, filter_type='sobel')



    def ed_filter(self,img_matrix,size, img_name,filter_type):
        file = FileIO()

        if (filter_type == 'prewitt'):
            horizontal_filter = np.array([[-1, 0, 1], [-1, 0, 1],[-1, 0, 1]])
            vertical_filter = np.array([[1, 1, 1], [0, 0, 0],[-1, -1, -1]])

        else:
            horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            vertical_filter = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

        img_horizontal = self.convolution(img_matrix,size[0],size[1],horizontal_filter)
        img_vertical = self.convolution(img_matrix,size[0],size[1],vertical_filter)

        combined_matrix = self.combine_edge_matrices(img_horizontal, img_vertical)
        combined_min, combined_max = self.get_min_max(combined_matrix, size[0], size[1])
        combined_matrix = self.normalize_matrix_int(combined_max, combined_min, combined_matrix, size[0], size[1])

        new_img_name = "ed_only_" + filter_type+ '_'  + img_name
        #flat_image = np.reshape(combined_matrix,(1,size[0]*size[1]))

        flat_image = combined_matrix.flatten()
        #print(flat_image)
        flat_image = flat_image.astype(np.int)
        flat_image = flat_image.tolist()
        #print(flat_image)
        #print(chr(flat_image[5]))
        flat_image = [chr(i) for i in flat_image]
        #print(flat_image)
        flat_image = "".join(flat_image)

        #print(flat_image)
        #flat_image = np.fromstring(flat_image, dtype=str)

        #print(flat_image)
        file.save_pgm(flat_image, size, new_img_name)

        threshold = self.get_global_threshold(combined_matrix, 255, 0)
        print("T: ", threshold)
        final_matrix = self.apply_threshold(combined_matrix, threshold)

        new_img_name = "ed_th_" + filter_type + '_' + img_name
        #flat_image = np.reshape(final_matrix, (1, size[0] * size[1]))

        flat_image = final_matrix.flatten()
        flat_image = flat_image.astype(np.int)
        flat_image = flat_image.tolist()
        flat_image = [chr(i) for i in flat_image]
        flat_image = "".join(flat_image)

        file.save_pgm(flat_image, size, new_img_name)


    def combine_edge_matrices(self, img_horizontal, img_vertical):
        combined_matrix = np.abs(img_horizontal) + np.abs(img_vertical)
        combined_matrix = combined_matrix.astype(np.int)
        #print(combined_matrix)
        return combined_matrix

    def get_global_threshold(self,img_matrix, img_max, img_min):
        t = (int)((img_max + img_min)/2)
        difference_limit = 2
        prev_t = t + difference_limit + 1
        smaller_count = 0
        smaller_sum = 0
        greater_count = 0
        greater_sum = 0

        while(abs(t - prev_t) >= 2):

            for line in img_matrix:
                for pix in line:
                    if pix >= t:
                        greater_count += 1
                        greater_sum += pix
                    else:
                        smaller_count += 1
                        smaller_sum += pix
            smaller_mean = (int)(smaller_sum/smaller_count)
            greater_mean = (int)(greater_sum/greater_count)
            prev_t = t
            t = (int)((smaller_mean + greater_mean)/2)
        return t


    def apply_threshold(self,img_matrix, threshold):

        for line in img_matrix:
            for pix in line:
                if pix >= threshold:
                    pix = 0
                else:
                    pix = 255

        return img_matrix



    def convolution(self, img_matrix, width, height, filter):
        offset = (int)((len(filter)-1)/2)
        print('offset: ',offset)
        filter_sum = 0
        filtered_matrix = np.empty(img_matrix.shape)

        for i in range(offset, width - offset):
            for j in range(offset, height - offset):
                filter_sum = self.apply_filter(img_matrix, i , j, filter, offset)
                filtered_matrix[i - offset][j - offset] = (int)(filter_sum/8)

        return filtered_matrix

    def apply_filter(self, img_matrix, i, j, filter, offset):
        filter_width = len(filter)
        filter_height = len(filter[0])
        filter_sum = 0
        i = i - offset
        j = j - offset

        for k in range(filter_width):
            for l in range(filter_height):
                filter_sum += (filter[k][l]*img_matrix[i + k][j + l])

        return filter_sum

    def normalize_matrix_int(self,max, min, matrix, width, height):

        for i in range(width):
            for j in range(height):
                 matrix[i][j] = ((matrix[i][j] - min) / (max - min)) * 255
                 # if x > 1:
                 matrix[i][j] = (int)(matrix[i][j])

        return matrix

    def get_min_max(self,matrix, width, height):
        max_val = 0
        min_val = 0

        for list in matrix:
            if max(list) > max_val:
                max_val = max(list)
            if min(list) < min_val:
                min_val = min(list)

        return min_val, max_val



class FileIO:
    def read_pgm_to_matrix(self,file_name):
        with open(file_name, 'r',  encoding="Latin-1") as file:
            print("Reading ",file_name)
            file.readline()
            size = file.readline().strip('\n').split(' ')
            size = list(map(int, size))
            print(size)
            file.readline()
            content = file.read().strip('\n')
            content = [ord(c) for c in content]

            content = np.array(list(content))
            content = np.reshape(content,(size[0], size[1]))
            print(content)
            print(content.__len__())
            print("\n")
            return content,size

    def save_pgm(self,flat_image, size, img_name):
        with open(img_name,"w+",  encoding="Latin-1") as f:
            f.write("P5\n")
            f.write("{} {}\n".format(size[0], size[1]))
            f.write("255\n")
            f.write(flat_image)


if (__name__ == '__main__'):
    ed = EdgeDetection()
    img_names = ['house.256.pgm', 'monarch.512.pgm', 'pentagone.1024.pgm']

    for name in img_names:
        ed.detect_edges(name)