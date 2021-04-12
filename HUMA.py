import cv2
import cv2 as cv
from matplotlib import pyplot as plt

def templateMatch(originalImagePath,templatePath):

    tempList1 = templatePath.split("/")
    tempList2 = tempList1[1].split(".")
    tempList3 = tempList2[0].split("_")
    templateName = tempList3[0] + " " + tempList3[1]

    # Fotograflar okundu.
    original_image = cv2.imread(originalImagePath)
    template_original = cv2.imread(templatePath)

    # Fotograflar gri skalaya cevrildi.
    imageGray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template_original, cv2.COLOR_BGR2GRAY)

    # Template in yükseklik ve genisligi
    h, w = templateGray.shape

    # Template Matching de kullanılan methodlar listelendi.
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    for meth in methods:
        img = imageGray.copy()
        method = eval(meth)

        #Template Matching uygulama
        res = cv2.matchTemplate(imageGray, templateGray, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        #TM_SQDIFF yada TM_SQDIFF_NORMED methodlarında minumum alinir.
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        # tespit edilen bölge dikdörtgenle cizildi.
        cv.rectangle(img, top_left, bottom_right, (255, 0, 0), 5)

        # tespit edilen bölge orjinal fotografta dikdörtgenle cizildi.
        cv2.rectangle(original_image, top_left, bottom_right, (222, 100, 35), 4)
        cv2.putText(original_image, templateName, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # plt.subplot komutuyla ekrana uygulanan methodlar ve sonuclari verildi.
        plt.subplot(221), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

        plt.subplot(223), plt.imshow(template_original, cmap='hsv')
        plt.title('Template'), plt.xticks([]), plt.yticks([])

        plt.suptitle(meth)
        plt.show()

    # orjinal resim gösterildi.
    cv2.imshow('Original', original_image)
    cv2.waitKey(0)

def main():
    originalImagePath,templatePath='pizza_resimleri/tavuklu_pizza.png',"templates/tavuklu_pizza_template.png"
    templateMatch(originalImagePath,templatePath)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()