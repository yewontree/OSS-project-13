import numpy as np
from settings import finger_tips
import logging, coloredlogs


def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def to_list_landmarks(self, hand_landmarks):
    """change hand_landmarks' position data to list, and return"""
    data = []
    for mark in hand_landmarks:
        data.append([mark.x, mark.y, mark.z])
    return data


def to_ndarray(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])


def radius(A, B, C):
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    radius = a * b * c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a * a * (b * b + c * c - a * a)
    b2 = b * b * (a * a + c * c - b * b)
    b3 = c * c * (a * a + b * b - c * c)
    center = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    center /= b1 + b2 + b3
    return radius if radius > 0 else np.inf


class Algorithm:  # algorithm class interface
    def detect_tip_finger(self, hand_landmarks):
        """returns tuple(wrist_point, tip_fingers)"""
        raise NotImplementedError

    def __init__(self):
        from utils import get_args

        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level=get_args().log_a.upper(), logger=self.logger)  # logger 설정, logger.debug() 함수로 로그메시지 표시

    def stats(self):
        pass


class Basic(Algorithm):
    def detect_tip_finger(self, hand_landmarks):
        wrist_point = hand_landmarks[0]
        dists = []
        for mark in hand_landmarks:
            dists.append(distance(wrist_point, mark))
        dists = np.array(dists)
        tips_dist = dists[finger_tips]
        tip_fingers = np.where(tips_dist > dists.mean() * 1.0, 1, 0)

        return wrist_point, np.array(tip_fingers)


class ThumbCheck(Algorithm):
    def detect_tip_finger(self, hand_landmarks):
        wrist_point = hand_landmarks[0]
        dists = []
        for mark in hand_landmarks:
            dists.append(distance(wrist_point, mark))
        dists = np.array(dists)

        tip_fingers = []

        # Thumb Check by Heuristic Algorithm
        palm_mean = dists[np.array(finger_tips[:-1]) + 1].mean()
        thumb1 = (dists[finger_tips[0]] - palm_mean) / palm_mean
        thumb2 = (dists[finger_tips[0]] - dists[finger_tips[0] - 1]) / dists[finger_tips[0] + 1]
        self.logger.debug(
            f"thumb1: {thumb1:>.2f}, thumb2: {thumb2:>.2f}, palm_mean: {palm_mean:>.2f}, finger_tip: {dists[finger_tips[0]]:>.2f}"
        )
        if thumb1 > 0.3 or thumb2 > 0.15:
            tip_fingers.append(1)
        else:
            tip_fingers.append(0)
        average = dists[5:].mean()
        tips_dist = dists[finger_tips[1:]]
        tip_fingers.extend(np.where(tips_dist > average * 1.0, 1, 0))

        return wrist_point, np.array(tip_fingers)


class Curvature(Algorithm):
    def detect_tip_finger(self, hand_landmarks):
        wrist_point = hand_landmarks[0]
        indices = [[2, 3, 4], [5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20]]
        fingers_radius = []
        for index in indices:
            marks = []
            for mark in hand_landmarks[index]:
                marks.append(to_ndarray(mark))
            fingers_radius.append(radius(*marks))

        self.logger.debug(f"radius: {fingers_radius}")

        return wrist_point, np.array([0, 0, 0, 0, 0])


# TODO  Add more algorithms ...


from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark, Landmark


# This data was collected from logger output for algorithm test
result = HandLandmarkerResult(
    handedness=[
        [Category(index=0, score=0.9763209819793701, display_name="Right", category_name="Right")],
        [Category(index=1, score=0.9860826730728149, display_name="Left", category_name="Left")],
    ],
    hand_landmarks=[
        [
            NormalizedLandmark(x=0.21064141392707825, y=0.9594322443008423, z=1.345023150634006e-07, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.2837805449962616, y=0.9113443493843079, z=-0.04489205405116081, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.3448058068752289, y=0.8382454514503479, z=-0.06692240387201309, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.38201597332954407, y=0.7709337472915649, z=-0.08703847974538803, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.4085865616798401, y=0.707824170589447, z=-0.10775782912969589, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.2968626022338867, y=0.661354124546051, z=-0.043923161923885345, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.31318265199661255, y=0.5601280927658081, z=-0.07620017230510712, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.3181244134902954, y=0.5040861368179321, z=-0.10431789606809616, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.3224550485610962, y=0.4472241997718811, z=-0.12673670053482056, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.23359636962413788, y=0.6474196910858154, z=-0.04797318950295448, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.2233859896659851, y=0.5254405736923218, z=-0.07917188853025436, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.21577979624271393, y=0.45171815156936646, z=-0.1072872206568718, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.20949514210224152, y=0.3849438428878784, z=-0.12989121675491333, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.17589671909809113, y=0.6706959009170532, z=-0.057498231530189514, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.15584523975849152, y=0.5568912029266357, z=-0.09102964401245117, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.1462581604719162, y=0.4848143756389618, z=-0.11542705446481705, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.14073675870895386, y=0.4150317311286926, z=-0.13399912416934967, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.12499107420444489, y=0.7222641706466675, z=-0.07132282853126526, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.09029051661491394, y=0.6474871039390564, z=-0.10664846748113632, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.06925828754901886, y=0.5948644280433655, z=-0.12516100704669952, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.05301786959171295, y=0.5394454598426819, z=-0.13764823973178864, visibility=0.0, presence=0.0),
        ],
        [
            NormalizedLandmark(x=0.9235824346542358, y=1.0309252738952637, z=1.9546399698811e-07, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.8356434106826782, y=1.0211056470870972, z=-0.03939593955874443, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.7545398473739624, y=0.9684244394302368, z=-0.06876949220895767, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.692676842212677, y=0.91277015209198, z=-0.09754594415426254, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.6305529475212097, y=0.8756945133209229, z=-0.12656623125076294, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.7527620196342468, y=0.7788661122322083, z=-0.04124521091580391, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.7173256278038025, y=0.6849173903465271, z=-0.07828783243894577, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.6928258538246155, y=0.6424496173858643, z=-0.10874152928590775, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.6667567491531372, y=0.6055691242218018, z=-0.1305210143327713, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.8115929365158081, y=0.7347450256347656, z=-0.04783811420202255, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.7830588221549988, y=0.6047554016113281, z=-0.0769067257642746, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.7611132264137268, y=0.531964123249054, z=-0.10150232166051865, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.7375922799110413, y=0.46349960565567017, z=-0.12158867716789246, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.87689608335495, y=0.7326712012290955, z=-0.060536906123161316, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.8603983521461487, y=0.6133061051368713, z=-0.09238626062870026, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.8447144031524658, y=0.5481432676315308, z=-0.11508900672197342, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.8242869973182678, y=0.48641738295555115, z=-0.13098853826522827, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.9431254863739014, y=0.7576470375061035, z=-0.07649501413106918, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.9547375440597534, y=0.6630425453186035, z=-0.10152021050453186, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.9624773859977722, y=0.603669285774231, z=-0.11481428891420364, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.96138596534729, y=0.5442957282066345, z=-0.12515658140182495, visibility=0.0, presence=0.0),
        ],
    ],
    hand_world_landmarks=[
        [
            Landmark(x=-0.013283854350447655, y=0.07887603342533112, z=0.05131107196211815, visibility=0.0, presence=0.0),
            Landmark(x=0.016448453068733215, y=0.0658477246761322, z=0.02409128099679947, visibility=0.0, presence=0.0),
            Landmark(x=0.039788052439689636, y=0.05029323697090149, z=-0.0003997418098151684, visibility=0.0, presence=0.0),
            Landmark(x=0.055975258350372314, y=0.030879897996783257, z=-0.03019706904888153, visibility=0.0, presence=0.0),
            Landmark(x=0.06512420624494553, y=0.0064541045576334, z=-0.04875006899237633, visibility=0.0, presence=0.0),
            Landmark(x=0.025771072134375572, y=-0.002051018178462982, z=-0.0028116374742239714, visibility=0.0, presence=0.0),
            Landmark(x=0.032458558678627014, y=-0.025493795052170753, z=-0.018725447356700897, visibility=0.0, presence=0.0),
            Landmark(x=0.035147443413734436, y=-0.039586473256349564, z=-0.029877914115786552, visibility=0.0, presence=0.0),
            Landmark(x=0.032964929938316345, y=-0.04886827617883682, z=-0.06215761601924896, visibility=0.0, presence=0.0),
            Landmark(x=0.00286497944034636, y=-0.005638806615024805, z=0.0033795118797570467, visibility=0.0, presence=0.0),
            Landmark(x=0.0034256586804986, y=-0.04106694459915161, z=-0.011029153130948544, visibility=0.0, presence=0.0),
            Landmark(x=-0.0006768573075532913, y=-0.05862870067358017, z=-0.03267596289515495, visibility=0.0, presence=0.0),
            Landmark(x=0.001859547570347786, y=-0.07198858261108398, z=-0.05828790366649628, visibility=0.0, presence=0.0),
            Landmark(x=-0.018321648240089417, y=-0.0015335357747972012, z=0.0014951116172596812, visibility=0.0, presence=0.0),
            Landmark(x=-0.020739689469337463, y=-0.03143496811389923, z=-0.011917880736291409, visibility=0.0, presence=0.0),
            Landmark(x=-0.02297375351190567, y=-0.04714013636112213, z=-0.030987491831183434, visibility=0.0, presence=0.0),
            Landmark(x=-0.02170654758810997, y=-0.06317432224750519, z=-0.05329478532075882, visibility=0.0, presence=0.0),
            Landmark(x=-0.03793773800134659, y=0.014064804650843143, z=0.0043105329386889935, visibility=0.0, presence=0.0),
            Landmark(x=-0.043975893408060074, y=-0.005810510367155075, z=-0.005668495316058397, visibility=0.0, presence=0.0),
            Landmark(x=-0.04799024760723114, y=-0.02365400828421116, z=-0.019594427198171616, visibility=0.0, presence=0.0),
            Landmark(x=-0.05266883224248886, y=-0.036627285182476044, z=-0.0382767878472805, visibility=0.0, presence=0.0),
        ],
        [
            Landmark(x=0.033120784908533096, y=0.07631296664476395, z=0.03888420760631561, visibility=0.0, presence=0.0),
            Landmark(x=0.00025022553745657206, y=0.06820909678936005, z=0.014341151341795921, visibility=0.0, presence=0.0),
            Landmark(x=-0.019546953961253166, y=0.055504098534584045, z=-0.0017344402149319649, visibility=0.0, presence=0.0),
            Landmark(x=-0.04485560208559036, y=0.0409419871866703, z=-0.02076880820095539, visibility=0.0, presence=0.0),
            Landmark(x=-0.06468529999256134, y=0.02064167708158493, z=-0.029397213831543922, visibility=0.0, presence=0.0),
            Landmark(x=-0.030952753499150276, y=0.009128730744123459, z=4.424931466928683e-05, visibility=0.0, presence=0.0),
            Landmark(x=-0.038678910583257675, y=-0.013321931473910809, z=-0.016489673405885696, visibility=0.0, presence=0.0),
            Landmark(x=-0.04717171564698219, y=-0.02779410593211651, z=-0.02945227175951004, visibility=0.0, presence=0.0),
            Landmark(x=-0.05647434666752815, y=-0.030472951009869576, z=-0.061893612146377563, visibility=0.0, presence=0.0),
            Landmark(x=-0.006923678796738386, y=-0.0035736323334276676, z=0.005222097504884005, visibility=0.0, presence=0.0),
            Landmark(x=-0.014377178624272346, y=-0.04074784368276596, z=-0.010445699095726013, visibility=0.0, presence=0.0),
            Landmark(x=-0.026670830324292183, y=-0.057290975004434586, z=-0.030795659869909286, visibility=0.0, presence=0.0),
            Landmark(x=-0.036718450486660004, y=-0.07431327551603317, z=-0.054011065512895584, visibility=0.0, presence=0.0),
            Landmark(x=0.019289521500468254, y=-0.008024852722883224, z=0.00022032318520359695, visibility=0.0, presence=0.0),
            Landmark(x=0.012678624130785465, y=-0.035582371056079865, z=-0.016720257699489594, visibility=0.0, presence=0.0),
            Landmark(x=0.004036860074847937, y=-0.05075448378920555, z=-0.039230551570653915, visibility=0.0, presence=0.0),
            Landmark(x=-0.0069150798954069614, y=-0.06278428435325623, z=-0.06176400184631348, visibility=0.0, presence=0.0),
            Landmark(x=0.03796424716711044, y=0.0022025457583367825, z=-0.0011515113292261958, visibility=0.0, presence=0.0),
            Landmark(x=0.042929913848638535, y=-0.020769817754626274, z=-0.007764710579067469, visibility=0.0, presence=0.0),
            Landmark(x=0.04473039507865906, y=-0.038350507616996765, z=-0.022298036143183708, visibility=0.0, presence=0.0),
            Landmark(x=0.040563520044088364, y=-0.050401072949171066, z=-0.04034033045172691, visibility=0.0, presence=0.0),
        ],
    ],
)


if __name__ == "__main__":
    hand_landmarks_list = result.hand_landmarks

    # TODO Test Algorithm here with had_landmarks_list
