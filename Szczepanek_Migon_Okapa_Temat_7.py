# Projekt TOM - Estymacja objętości wątroby ze skanów CT
# Natalia Szczepanek, Jan Migoń, Maciej Okapa

# Zaimportowanie potrzebnych bibliotek

from medpy.io import load
import numpy as np
import cv2
from matplotlib import pyplot as plt
import queue
from skimage import color
from skimage import measure


# Pobranie danych przy pomocy funkcji load z biblioteki medpy.io

patient_01_data, patient_01_header = load('/Users/nszczepanek/Desktop/projekt/Patient01.mha')
patient_02_data, patient_02_header = load('/Users/nszczepanek/Desktop/projekt/Patient02.mha')
patient_03_data, patient_03_header = load('/Users/nszczepanek/Desktop/projekt/Patient03.mha')
patient_04_data, patient_04_header = load('/Users/nszczepanek/Desktop/projekt/Patient04.mha')
patient_05_data, patient_05_header = load('/Users/nszczepanek/Desktop/projekt/Patient05.mha')


#************************************************************************************************************
# region_growing_local()
# opis: funkcja realizująca lokalny algorytm rozrostu obszarów
# parametry : image - przkerój, seed - punkt początkowy, threshold_diff - różnica w poziomie intensywności
# wartości zwracane - obraz po realizacji lokalnego algorytmu rozrostu obszarów, wysegmentowany
#************************************************************************************************************


def region_growing_local(image, seed, threshold_diff):
	y_size, x_size = image.shape
	output_image = np.zeros(image.shape)
	seed_intensity = image[seed]
	uth = seed_intensity + threshold_diff; dth = seed_intensity - threshold_diff

	def get_neighbours(coordinate):
		neighbours = []
		possible_steps = [-1, 0, 1]
		for i in possible_steps:
			for j in possible_steps:
				current_y = min(max(coordinate[0] + j, 0), y_size - 1)
				current_x = min(max(coordinate[1] + i, 0), x_size - 1)
				neighbours.append((current_y, current_x))
		return neighbours

	our_queue = queue.Queue()
	visited = set()
	our_queue.put(seed)

	while not our_queue.empty():
		current_item = our_queue.get()
		current_neighbours = get_neighbours(current_item)
		if image[current_item] < uth and image[current_item] > dth:
			output_image[current_item] = 1

		for neighbour in current_neighbours:
			if neighbour in visited:
				continue
			else:
				visited.add(neighbour)
				if image[neighbour] < uth and image[neighbour] > dth:
					our_queue.put(neighbour)
	return output_image


#************************************************************************************************************
# region_growing_global()
# opis: funkcja realizująca globalny algorytm rozrostu obszarów
# parametry : image - przkerój, seed - punkt początkowy, threshold_diff - różnica w poziomie intensywności
# wartości zwracane - obraz po realizacji globalnego algorytmu rozrostu obszarów, wysegmentowany
#************************************************************************************************************


def region_growing_global(image, seed, threshold_diff):
	seed_intensity = image[seed]
	uth = seed_intensity + threshold_diff; dth = seed_intensity - threshold_diff
	thresholed_image = np.logical_and(image < uth, image > dth)
	labeled_image = measure.label(thresholed_image, background=0)
	indices = labeled_image == labeled_image[seed]
	output_image = np.zeros(image.shape)
	output_image[indices] = 1
	return output_image



# stworzenie pustej listy na wysegmentowane dane pacjenta
# poniższy kod jest uniwersalny dla każdego wprowadzonego pacjenta
# różnica - konieczna zmiana jedynie w punkcie początkowym tj. seed



patient_1_list=[]
for i in range(3,27):    #wybrane zostaly przekroje z widoczną wątrobą, stąd range
    image=np.uint8(patient_01_data[:,:,i].T) #rzutowanie na typ uint8
    image=color.rgb2gray(image) #zmimana rgb na skalę szarości
    plt.figure() #wyswietlenie wszystkich przekroi
    plt.imshow(image,cmap='gray')
    ret,thresh1 = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #wykonanie segmentacji przy uzyciu algorytmu Otsu
    #plt.figure()
    #plt.imshow(thresh1,cmap='gray')
    #print(np.max(thresh1))
    image_norm = (thresh1 - np.min(thresh1)) / (np.max(thresh1) - np.min(thresh1)) #normalizacja wartosci
    plt.figure()
    plt.imshow(image_norm,cmap='gray')
    #print(np.max(image_norm))

    #stworzenie maski 5x5
    kernel = np.ones((5,5),np.uint8)
    #wykonanie erozji (dwukrotnie)
    erosion = cv2.erode(image_norm,kernel,iterations = 2)
    plt.figure()
    plt.imshow(erosion,cmap='gray')
    #wykonanie funkcji otwarcia - tj dylatacja z erozji dwukrotnie
    opening=cv2.dilate(erosion,kernel,iterations=2)
    plt.figure()
    plt.imshow(opening,cmap='gray')


    seed = (240,70) #wprowadzenie punktu początkowego
    region_growing_global_image = region_growing_global(opening, seed, 0.0001)     #zastosowanie algorytmu roztostu obszaru
    patient_1_list.append(region_growing_global_image)
    plt.figure()
    plt.imshow(region_growing_global_image, cmap='gray')
    plt.title(str(i))


# Zliczenie wartości białych pikseli w celu obliczenia objętości


white_pix=[]
for i,image in enumerate (patient_1_list):
    suma=np.sum(image)
    white_pix.append(suma)
    print(i+3,suma)


# Pobranie informacji na temat voxeli


distance_01 = patient_01_header.get_voxel_spacing()
distance_01


# Obliczenie objętości w cm^3


volume=sum(white_pix)*distance_01[0]**2*distance_01[2]
volume/1000


# Pokazanie zdjęć innego pacjenta, tj. przekroi z widoczną wątrobą


for i in range(47,83,3):
    plt.figure()
    plt.imshow(patient_02_data[:,:,i].T, cmap = 'gray')
    plt.title(str(i))


# Transponowanie przekroi i uwidocznienie danych


plt.imshow(patient_02_data[:,:,47].T[200:300,70:150],cmap='gray')


# Dalsze operacje są analogiczne do powyższych, jedyna zmiana to zmiana pacjenta i przekrojów


from skimage import filters

image=np.uint8(patient_02_data[:,:,50].T)
image=color.rgb2gray(image)
val = filters.threshold_otsu(image[150:300,50:150])
val


# W celu usprawnienia algorytmu i ułatwienia działania programu, należałoby stworzyć listę, która przechowuje wszystkie seedy oraz tuple z informacją o
# numerach przekrojów, na których widoczna była wątroba i odpowiednie dopasowanie w zależności od tego, którego pacjenta chcemy stosować
# Można wprowadzić interaktywny wybór, co ułatwiłoby i umożliwiłoby sprawną pracę lekarzowi


image=np.uint8(patient_02_data[:,:,55].T)
image=color.rgb2gray(image)
plt.figure()
plt.imshow(image,cmap='gray')
ret,thresh1 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#plt.figure()
#plt.imshow(thresh1,cmap='gray')
#print(np.max(thresh1))
image_norm = (thresh1 - np.min(thresh1)) / (np.max(thresh1) - np.min(thresh1))
plt.figure()
plt.imshow(image_norm,cmap='gray')
#print(np.max(image_norm))
kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(image_norm,kernel,iterations = 4)
plt.figure()
plt.imshow(erosion,cmap='gray')
opening=cv2.dilate(erosion,kernel,iterations=1)
plt.figure()
plt.imshow(opening,cmap='gray')






