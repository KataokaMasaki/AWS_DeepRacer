# -*- coding: utf-8 -*-

import math
# インポートエラーとなるため、コメントアウト
# import traceback

"""
This is the source code you cut and paste into AWS console. It consists of RewardEvaluator class that is instantiated
by the code of the desired reward_function(). The  RewardEvaluator contains a set of elementary  "low level" functions 
 for example the distance calculation between waypoints, directions as well as higher-level functions (e.g. nearest turn 
direction and distance) allowing you to design more complex reward logic.

AWSのコンソールにカット＆ペーストするソースコードです。RewardEvaluatorクラスがインスタンス化されています。
のコードによってインスタンス化されます。RewardEvaluatorには、基本的な「低レベル」関数のセットが含まれています。
 例えば、ウェイポイント間の距離計算、方向、そしてより高度な関数(例えば、最も近いターンの方向と距離)などです。
方向と距離) があり、より複雑な報酬ロジックを設計することができます。
"""


class RewardEvaluator:

    # CALCULATION CONSTANTS - change for the performance fine tuning

    # Define minimum and maximum expected speed interval for the training. Both values should be corresponding to
    # parameters you are going to use for the Action space. Set MAX_SPEED equal to maximum speed defined there,
    # MIN_SPEED should be lower (just a bit) then expected minimum defined speed (e.g. Max speed set to 5 m/s,
    # speed granularity 3 => therefore, MIN_SPEED should be less than 1.66 m/s.
    # トレーニングに必要な最低・最高速度間隔を定義します. どちらの値も,
    # アクションスペースで使用するパラメータに対応するものである必要があります. MAX_SPEEDはそこで定義された最高速度と等しくなるように設定してください,
    # MIN_SPEEDは, 定義された最低速度の予想値よりも(ほんの少しだけ)低い値にします(例. Max speed set to 5 m/s,
    # 速度粒度 3 => したがって, MIN_SPEED は 1.66 m/s よりも小さくなければならない.
    MAX_SPEED = float(5.0)
    MIN_SPEED = float(1.5)

    # Define maximum steering angle according to the Action space settings. Smooth steering angle threshold is used to
    # set a steering angle still considered as "smooth". The value must be higher than minimum steering angle determined
    # by the steering Action space. E.g Max steering 30 degrees, granularity 3 => SMOOTH_STEERING_ANGLE_TRESHOLD should
    # be higher than 10 degrees.
    # まだ "滑らか "であるとみなされるステアリング角度を設定する。この値は, ステアリングアクションスペースで設定された
    # 最小ステアリング角よりも大きくなければならない.
    # ステアリング操作空間によって決定される最小ステアリング角よりも高い値でなければならない. 例: 最大ステアリング角30度, 粒度3 => SMOOTH_STEERING_ANGLE_TRESHOLDは10度以上でなければならない.
    # 10度以上でなければならない.
    MAX_STEERING_ANGLE = 30
    SMOOTH_STEERING_ANGLE_TRESHOLD = 15  # Greater than minimum angle defined in action space

    # Constant value used to "ignore" turns in the corresponding distance (in meters). The car is supposed to drive
    # at MAX_SPEED (getting a higher reward). In case within the distance is a turn, the car is rewarded when slowing
    # down.
    # Action space の設定に従って、最大ステアリング角を定義します。スムーズステアリングアングルの閾値を設定することで
    SAFE_HORIZON_DISTANCE = 0.8  # meters, able to fully stop. See ANGLE_IS_CURVE.

    # Constant to define accepted distance of the car from the center line.
    # 車の中心線からの許容距離を定義する定数.
    CENTERLINE_FOLLOW_RATIO_TRESHOLD = 0.12

    # Constant to define a threshold (in degrees), representing max. angle within SAFE_HORIZON_DISTANCE. If the car is
    # supposed to start steering and the angle of the farthest waypoint is above the threshold, the car is supposed to
    # slow down
    # SAFE_HORIZON_DISTANCE 内の最大角度を示す閾値を定義する定数(度単位)です。クルマがステアリングを開始すると仮定した場合
    # ステアリングを開始しようとした時に、最も遠いウェイポイントの角度が閾値を超えている場合、 # 車は減速するようになっています。
    # 速度を落とす
    ANGLE_IS_CURVE = 3

    # A range the reward value must fit in.
    # 報酬の値が収まるべき範囲.
    PENALTY_MAX = 0.001
    REWARD_MAX = 89999  # 100000

    # params is a set of input values provided by the DeepRacer environment. For each calculation
    # this is provided
    # params は、DeepRacer 環境から提供される入力値のセットです。各計算についてこれが提供されます
    params = None

    # Class properties - status values extracted from "params" input
    # クラスのプロパティ - "params "入力から抽出されたステータス値
    all_wheels_on_track = None
    x = None
    y = None
    distance_from_center = None
    is_left_of_center = None
    is_reversed = None
    heading = None
    progress = None
    steps = None
    speed = None
    steering_angle = None
    track_width = None
    waypoints = None
    closest_waypoints = None
    nearest_previous_waypoint_ind = None
    nearest_next_waypoint_ind = None

    log_message = ""

    # method used to extract class properties (status values) from input "params"
    # 入力 "params" からクラスのプロパティ (ステータス値) を抽出するために用いられるメソッド。
    def init_self(self, params):
        self.all_wheels_on_track = params['all_wheels_on_track']
        self.x = params['x']
        self.y = params['y']
        self.distance_from_center = params['distance_from_center']
        self.is_left_of_center = params['is_left_of_center']
        self.is_reversed = params['is_reversed']
        self.heading = params['heading']
        self.progress = params['progress']
        self.steps = params['steps']
        self.speed = params['speed']
        self.steering_angle = params['steering_angle']
        self.track_width = params['track_width']
        self.waypoints = params['waypoints']
        self.closest_waypoints = params['closest_waypoints']
        self.nearest_previous_waypoint_ind = params['closest_waypoints'][0]
        self.nearest_next_waypoint_ind = params['closest_waypoints'][1]

    # RewardEvaluator Class constructor
    # RewardEvaluator クラスのコンストラクタ
    def __init__(self, params):
        self.params = params
        self.init_self(params)

    # Method used to "print" status values and logged messages into AWS log. Be aware of additional cost Amazon will
    # charge you when logging is used heavily!!!
    # ステータス値やログメッセージをAWSログに "印刷 "するために使用される方法。ロギングを多用する場合、Amazonから追加コストが発生することに注意してください。
    # ロギングが多用される場合、Amazonはあなたに追加コストを請求します!!!
    def status_to_string(self):
        status = self.params
        if 'waypoints' in status: del status['waypoints']
        status['debug_log'] = self.log_message
        print(status)

    # Gets ind'th waypoint from the list of all waypoints retrieved in params['waypoints']. Waypoints are circuit track
    # specific (every time params is provided it is same list for particular circuit). If index is out of range (greater
    # than len(params['waypoints']) a waypoint from the beginning of the list ir returned.
    # params['waypoints'] で取得した全てのウェイポイントの中から、ind番目のウェイポイントを取得します。ウェイポイントはサーキットトラック
    # (params が与えられる度に, 特定の回路に対応した同じリストが返されます). indexが範囲外(len(params['waypoint'])よりも大きい)の場合
    # indexが範囲外だった場合(len(params['waypoints'])、リストの先頭からのウェイポイントが返されます.
    def get_way_point(self, index_way_point):
        if index_way_point > (len(self.waypoints) - 1):
            return self.waypoints[index_way_point - (len(self.waypoints))]
        elif index_way_point < 0:
            return self.waypoints[len(self.waypoints) + index_way_point]
        else:
            return self.waypoints[index_way_point]

    # Calculates distance [m] between two waypoints [x1,y1] and [x2,y2]
    # 2つのウェイポイント [x1,y1] と [x2,y2] の間の距離 [m] を計算します。
    @staticmethod
    def get_way_points_distance(previous_waypoint, next_waypoint):
        return math.sqrt(pow(next_waypoint[1] - previous_waypoint[1], 2) + pow(next_waypoint[0] - previous_waypoint[0], 2))

    # 0 to -180 degrees, anti clockwise 0 to +180 degrees
    # Calculates heading direction between two waypoints - angle in cartesian layout. Clockwise values
    # 0度から-180度、反時計回り 0度から+180度
    # 2つのウェイポイント間の方位角を計算します - 直交座標系での角度です。時計回りの値
    @staticmethod
    def get_heading_between_waypoints(previous_waypoint, next_waypoint):
        track_direction = math.atan2(next_waypoint[1] - previous_waypoint[1], next_waypoint[0] - previous_waypoint[0])
        return math.degrees(track_direction)

    # Calculates the misalignment of the heading of the car () compared to center line of the track (defined by previous and
    # the next waypoint (the car is between them)
    # トラックの中心線に対する車の方位()のズレを計算します(前のウェイポイントと次のウェイポイントによって定義されます)
    # 次のウェイポイント(車はそれらの間にある)
    def get_car_heading_error(self):  # track direction vs heading
        next_point = self.get_way_point(self.closest_waypoints[1])
        prev_point = self.get_way_point(self.closest_waypoints[0])
        track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        track_direction = math.degrees(track_direction)
        return track_direction - self.heading

    # Based on CarHeadingError (how much the car is misaligned with th direction of the track) and based on the "safe
    # horizon distance it is indicating the current speed (params['speed']) is/not optimal.
    # CarHeadingError(車体がトラックの方向とどれだけずれているか)と「安全な」水平距離に基づいて、現在の速度(param[:speed])が最適かどうかを示します。
    # 水平距離に基づいて, 現在の速度 (params['speed']) が最適である/ないことを示します.
    def get_optimum_speed_ratio(self):
        if abs(self.get_car_heading_error()) >= self.MAX_STEERING_ANGLE:
            return float(0.34)
        if abs(self.get_car_heading_error()) >= (self.MAX_STEERING_ANGLE * 0.75):
            return float(0.67)
        current_position_xy = (self.x, self.y)
        current_wp_index = self.closest_waypoints[1]
        length = self.get_way_points_distance((self.x, self.y), self.get_way_point(current_wp_index))
        current_track_heading = self.get_heading_between_waypoints(self.get_way_point(current_wp_index),
                                                                   self.get_way_point(current_wp_index + 1))
        while True:
            from_point = self.get_way_point(current_wp_index)
            to_point = self.get_way_point(current_wp_index + 1)
            length = length + self.get_way_points_distance(from_point, to_point)
            if length >= self.SAFE_HORIZON_DISTANCE:
                heading_to_horizont_point = self.get_heading_between_waypoints(self.get_way_point(self.closest_waypoints[1]), to_point)
                if abs(current_track_heading - heading_to_horizont_point) > (self.MAX_STEERING_ANGLE * 0.5):
                    return float(0.33)
                elif abs(current_track_heading - heading_to_horizont_point) > (self.MAX_STEERING_ANGLE * 0.25):
                    return float(0.66)
                else:
                    return float(1.0)
            current_wp_index = current_wp_index + 1

    # Calculates angle of the turn the car is right now (degrees). It is angle between previous and next segment of the
    # track (previous_waypoint - closest_waypoint and closest_waypoint - next_waypoint)
    # 車が今回っている角度を計算します(度)。これは, トラックの前のセグメントと次のセグメントの間の角度である.
    # トラックの前のセグメントと次のセグメントの間の角度 (前方点 - 最寄り方位点, 最寄り方位点 - 次方位点) を計算する.
    def get_turn_angle(self):
        current_waypoint = self.closest_waypoints[0]
        angle_ahead = self.get_heading_between_waypoints(self.get_way_point(current_waypoint),
                                                         self.get_way_point(current_waypoint + 1))
        angle_behind = self.get_heading_between_waypoints(self.get_way_point(current_waypoint - 1),
                                                          self.get_way_point(current_waypoint))
        result = angle_ahead - angle_behind
        if angle_ahead < -90 and angle_behind > 90:
            return 360 + result
        elif result > 180:
            return -180 + (result - 180)
        elif result < -180:
            return 180 - (result + 180)
        else:
            return result

    # Indicates the car is in turn
    # 車が回転していることを示す
    def is_in_turn(self):
        if abs(self.get_turn_angle()) >= self.ANGLE_IS_CURVE:
            return True
        else:
            return False
        return False

    # Indicates the car has reached final waypoint of the circuit track
    # サーキットコースの最終ウェイポイントに到達したことを示します。
    def reached_target(self):
        max_waypoint_index = len(self.waypoints) - 1
        if self.closest_waypoints[1] == max_waypoint_index:
            return True
        else:
            return False

    # Provides direction of the next turn in order to let you reward right position to the center line (before the left
    # turn position of the car sligthly right can be rewarded (and vice versa) - see is_in_optimized_corridor()
    # 次のターンの方向を提供し、センターラインに対して右側の位置が報酬として与えられるようにする (左側の位置が報酬として与えられる前に)
    # 少し右に曲がる車の位置が報われる (逆も同様) - is_in_optimized_corridor() を参照
    def get_expected_turn_direction(self):
        current_waypoint_index = self.closest_waypoints[1]
        length = self.get_way_points_distance((self.x, self.y), self.get_way_point(current_waypoint_index))
        while True:
            from_point = self.get_way_point(current_waypoint_index)
            to_point = self.get_way_point(current_waypoint_index + 1)
            length = length + self.get_way_points_distance(from_point, to_point)
            if length >= self.SAFE_HORIZON_DISTANCE * 4.5:
                result = self.get_heading_between_waypoints(self.get_way_point(self.closest_waypoints[1]), to_point)
                if result > 2:
                    return "LEFT"
                elif result < -2:
                    return "RIGHT"
                else:
                    return "STRAIGHT"
            current_waypoint_index = current_waypoint_index + 1

    # Based on the direction of the next turn it indicates the car is on the right side to the center line in order to
    # drive through smoothly - see get_expected_turn_direction().
    # 次のターンの方向から、車がセンターラインに対して右側にあることを示し、スムーズに走り抜けられるようにします。
    # スムーズに走り抜けるために、車が中心線に対して右側にあることを示す - get_expected_turn_direction() を参照。
    def is_in_optimized_corridor(self):
        if self.is_in_turn():
            turn_angle = self.get_turn_angle()
            if turn_angle > 0:  # Turning LEFT - better be by left side
                if (self.is_left_of_center == True and self.distance_from_center <= (
                        self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width) or
                        self.is_left_of_center == False and self.distance_from_center <= (
                                self.CENTERLINE_FOLLOW_RATIO_TRESHOLD / 2 * self.track_width)):
                    return True
                else:
                    return False
            else:  # Turning RIGHT - better be by right side
                if self.is_left_of_center == True and self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD / 2 * self.track_width) or self.is_left_of_center == False and self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width):
                    return True
                else:
                    return False
        else:
            next_turn = self.get_expected_turn_direction()
            if next_turn == "LEFT":  # Be more righ side before turn
                if self.is_left_of_center == True and self.distance_from_center <= (
                        self.CENTERLINE_FOLLOW_RATIO_TRESHOLD / 2 * self.track_width) or self.is_left_of_center == False and self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width):
                    return True
                else:
                    return False
            elif next_turn == "RIGHT":  # Be more left side before turn:
                if self.is_left_of_center == True and self.distance_from_center <= (
                        self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width) or self.is_left_of_center == False and self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD / 2 * self.track_width):
                    return True
                else:
                    return False
            else:  # Be aligned with center line:
                if self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width):
                    return True
                else:
                    return False

    def is_optimum_speed(self):
        if abs(self.speed - (self.get_optimum_speed_ratio() * self.MAX_SPEED)) < (self.MAX_SPEED * 0.15) and self.MIN_SPEED <= self.speed <= self.MAX_SPEED:
            return True
        else:
            return False

    # Accumulates all logging messages into one string which you may need to write to the log (uncomment line
    # self.status_to_string() in evaluate() if you want to log status and calculation outputs.
    # 全てのロギングメッセージを一つの文字列にまとめ、ログに書き込む必要があるかもしれません (行のコメントを外します)
    # ステータスや計算結果を記録したい場合は、evaluate() の # self.status_to_string() をアンコメントしてください。
    def log_feature(self, message):
        if message is None:
            message = 'NULL'
        self.log_message = self.log_message + str(message) + '|'

    # Here you can implement your logic to calculate reward value based on input parameters (params) and use
    # implemented features (as methods above)
    # ここでは, 入力されたパラメータ(params)に基づいて報酬値を計算するロジックを実装し, # 以下の機能を利用することができます.
    # (上記のメソッドと同様に)実装されている機能
    def evaluate(self):
        self.init_self(self.params)
        result_reward = float(0.001)
        try:
            # No reward => Fatal behaviour, NOREWARD!  (out of track, reversed, sleeping)
            # 報酬なし⇒致命的な行動、NOREWARD!  (コースアウト、逆走、睡眠)
            if self.all_wheels_on_track == False or self.is_reversed == True or (self.speed < (0.1 * self.MAX_SPEED)):
                self.log_feature("all_wheels_on_track or is_reversed issue")
                self.status_to_string()
                return float(self.PENALTY_MAX)

            # REWARD 50 - EARLY Basic learning => easy factors accelerate learning
            # Right heading, no crazy steering
            # 50の報酬 - EARLY 基本的な学習⇒簡単な要因で学習が加速する
            # 正しい方向へ、おかしな舵取りをしない
            if abs(self.get_car_heading_error()) <= self.SMOOTH_STEERING_ANGLE_TRESHOLD:
                self.log_feature("getCarHeadingOK")
                result_reward = result_reward + self.REWARD_MAX * 0.3

            if abs(self.steering_angle) <= self.SMOOTH_STEERING_ANGLE_TRESHOLD:
                self.log_feature("getSteeringAngleOK")
                result_reward = result_reward + self.REWARD_MAX * 0.15

            # REWARD100 - LATER ADVANCED complex learning
            # Ideal path, speed wherever possible, carefully in corners
            # REWARD100 - LATER ADVANCED 複合学習
            # 理想的な道、可能な限りスピード、コーナーは慎重に
            if self.is_in_optimized_corridor():
                self.log_feature("is_in_optimized_corridor")
                result_reward = result_reward + float(self.REWARD_MAX * 0.45)

            if not (self.is_in_turn()) and (abs(self.speed - self.MAX_SPEED) < (0.1 * self.MAX_SPEED)) \
                    and abs(self.get_car_heading_error()) <= self.SMOOTH_STEERING_ANGLE_TRESHOLD:
                self.log_feature("isStraightOnMaxSpeed")
                result_reward = result_reward + float(self.REWARD_MAX * 1)

            if self.is_in_turn() and self.is_optimum_speed():
                self.log_feature("isOptimumSpeedinCurve")
                result_reward = result_reward + float(self.REWARD_MAX * 0.6)

            # REWAR - Progress bonus
            TOTAL_NUM_STEPS = 150
            if (self.steps % 100 == 0) and self.progress > (self.steps / TOTAL_NUM_STEPS):
                self.log_feature("progressingOk")
                result_reward = result_reward + self.REWARD_MAX * 0.4

            # Reach Max Waypoint - get extra reward
            # 最大ウェイポイントに到達すると、追加報酬を得ることができます。
            if self.reached_target():
                self.log_feature("reached_target")
                result_reward = float(self.REWARD_MAX)

        except Exception as e:
            print("Error : " + str(e))
            # print(traceback.format_exc())

        # Finally - check reward value does not exceed maximum value
        # 最後に - 報酬の値が最大値(100,000)を超えていないことを確認します。
        # if result_reward > 900000:
        #     result_reward = 900000
        if result_reward > 90000:
            result_reward = 90000

        self.log_feature(result_reward)
        # self.status_to_string()

        return float(result_reward)


"""
This is the core function called by the environment to calculate reward value for every point of time of the training. 
params: input values for the reward calculation (see above)

Usually, this function contains all reward calculations a logic implemented. Instead, this code example is instantiating 
RewardEvaluator which has implemented a set of features one can easily combine and use.

環境から呼び出されるコア関数で、トレーニングの各時点での報酬値を計算します。
params: 報酬計算のための入力値 (上記参照)

通常、この関数にはロジックが実装されたすべての報酬計算が含まれています。その代わり、このコード例では 
RewardEvaluator をインスタンス化し、簡単に組み合わせて使える機能を実装しています。
"""


def reward_function(params):
    re = RewardEvaluator(params)
    return float(re.evaluate())
