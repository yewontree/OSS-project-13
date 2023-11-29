import cv2


def main():
    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()
        if success:
            cv2.imshow(f"Finger Counter", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print(f"cap.read() error")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
