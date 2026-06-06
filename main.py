import cv2 as cv2
import matplotlib.pyplot as plt 

ow1 = cv2.imread("assets/oneway1.png", cv2.IMREAD_GRAYSCALE)
ow2 = cv2.imread("assets/oneway2.png", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create(nfeatures=50)

kp1, des1 = sift.detectAndCompute(ow1, None)
kp2, des2 = sift.detectAndCompute(ow2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1 , des2 , k=2)
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
result = cv2.drawMatches(
    ow1,
    kp1,
    ow2,
    kp2,
    good_matches,
    None
)

plt.figure(figsize=(15,8))
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.show()