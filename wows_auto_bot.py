"""
战舰世界自动游戏脚本
功能：自动选择战船、开始游戏、寻敌、瞄准、射击、检测被击毁、退出并开始下一局

@author seven
@since 2024-12-19
"""

import cv2
import numpy as np
import pyautogui
import time
import logging
from typing import Tuple, List, Optional, Dict
import sys
import random
from datetime import datetime
import os
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wows_auto_bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 配置pyautogui安全设置
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


class GameState(Enum):
    """游戏状态枚举"""
    PORT = "port"  # 港口界面
    LOADING = "loading"  # 加载中
    IN_BATTLE = "in_battle"  # 战斗中
    MAP_OPEN = "map_open"  # 地图打开
    DESTROYED = "destroyed"  # 被击毁
    BATTLE_RESULT = "battle_result"  # 战斗结算
    UNKNOWN = "unknown"  # 未知状态


class WowsAutoBot:
    """战舰世界自动游戏机器人"""
    
    # 屏幕分辨率
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    
    # 颜色范围定义（HSV颜色空间）
    RED_LOWER_1 = np.array([0, 100, 100])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([170, 100, 100])
    RED_UPPER_2 = np.array([180, 255, 255])
    
    WHITE_LOWER = np.array([0, 0, 200])
    WHITE_UPPER = np.array([180, 30, 255])
    
    # 鼠标移动参数
    MIN_MOVE_DURATION = 0.3
    MAX_MOVE_DURATION = 0.8
    MIN_RANDOM_DELAY = 0.1
    MAX_RANDOM_DELAY = 0.3
    
    # 游戏操作参数
    AIM_CHECK_INTERVAL = 0.5  # 瞄准检查间隔（秒）
    SHOOT_COOLDOWN = 2.0  # 射击冷却时间（秒）
    ENEMY_SEARCH_INTERVAL = 1.0  # 寻敌间隔（秒）
    
    def __init__(self):
        """初始化自动游戏机器人"""
        logger.info("初始化战舰世界自动游戏机器人")
        self.current_state = GameState.UNKNOWN
        self.player_pos: Optional[Tuple[int, int]] = None
        self.enemy_positions: List[Tuple[int, int]] = []
        self.target_enemy: Optional[Tuple[int, int]] = None
        self.last_shot_time = 0.0
        self.battle_count = 0
        self.running = True
        
    def random_delay(self, min_seconds: float = None, max_seconds: float = None) -> None:
        """
        随机延迟，模拟人类操作
        
        @param min_seconds: 最小延迟时间（秒）
        @param max_seconds: 最大延迟时间（秒）
        """
        if min_seconds is None:
            min_seconds = self.MIN_RANDOM_DELAY
        if max_seconds is None:
            max_seconds = self.MAX_RANDOM_DELAY
        
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
        logger.debug(f"随机延迟: {delay:.3f}秒")
    
    def capture_screen(self) -> np.ndarray:
        """
        捕获当前屏幕截图
        
        @return: 屏幕截图的numpy数组（BGR格式）
        @raise Exception: 当截图失败时抛出异常
        """
        try:
            screenshot = pyautogui.screenshot()
            img_array = np.array(screenshot)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            logger.error(f"捕获屏幕截图失败: {str(e)}")
            raise
    
    def detect_game_state(self, img: np.ndarray) -> GameState:
        """
        检测当前游戏状态
        
        @param img: 输入图像（BGR格式）
        @return: 游戏状态
        @raise Exception: 当检测失败时抛出异常
        """
        try:
            # 转换为HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 检测"加入战斗"按钮（红色按钮，通常在屏幕中央上方）
            # 检查中央上方区域是否有红色按钮
            center_region = img[100:200, 800:1120]  # 中央上方区域
            center_hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
            red_mask1 = cv2.inRange(center_hsv, self.RED_LOWER_1, self.RED_UPPER_1)
            red_mask2 = cv2.inRange(center_hsv, self.RED_LOWER_2, self.RED_UPPER_2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_pixels = cv2.countNonZero(red_mask)
            
            if red_pixels > 500:  # 如果有很多红色像素，可能是"加入战斗"按钮
                logger.info("检测到港口界面（加入战斗按钮）")
                return GameState.PORT
            
            # 检测"离开战斗"或"您的战舰已被击毁"文字
            # 检查底部中央区域
            bottom_region = img[900:1080, 600:1320]
            gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
            # 简单的文字检测：查找高对比度区域
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(binary)
            
            if white_pixels > 10000:  # 如果有大量白色像素，可能是被击毁界面
                logger.info("检测到被击毁界面")
                return GameState.DESTROYED
            
            # 检测地图是否打开（检查是否有网格）
            # 地图通常有网格线和坐标
            map_region = img[0:1080, 0:1920]
            map_gray = cv2.cvtColor(map_region, cv2.COLOR_BGR2GRAY)
            # 检测直线（网格线）
            edges = cv2.Canny(map_gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 20:  # 如果有很多直线，可能是地图
                logger.info("检测到地图界面")
                return GameState.MAP_OPEN
            
            # 检测战斗中（有瞄准镜、UI元素等）
            # 检查是否有绿色瞄准镜（中央区域）
            center_screen = img[400:680, 800:1120]
            center_hsv = cv2.cvtColor(center_screen, cv2.COLOR_BGR2HSV)
            green_lower = np.array([50, 100, 100])
            green_upper = np.array([70, 255, 255])
            green_mask = cv2.inRange(center_hsv, green_lower, green_upper)
            green_pixels = cv2.countNonZero(green_mask)
            
            if green_pixels > 100:  # 如果有绿色像素，可能是瞄准镜
                logger.info("检测到战斗界面")
                return GameState.IN_BATTLE
            
            logger.warning("无法确定游戏状态，返回未知状态")
            return GameState.UNKNOWN
            
        except Exception as e:
            logger.error(f"检测游戏状态失败: {str(e)}")
            return GameState.UNKNOWN
    
    def detect_red_enemies(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """
        识别图像中的红色多边形敌人
        
        @param img: 输入图像（BGR格式）
        @return: 敌人位置列表
        @raise Exception: 当识别失败时抛出异常
        """
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            mask1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
            mask2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            enemy_positions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 30:
                    continue
                
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = len(approx)
                
                if vertices < 3 or vertices > 6:
                    continue
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity > 0.7:
                            enemy_positions.append((cx, cy))
            
            return enemy_positions
            
        except Exception as e:
            logger.error(f"识别红色敌人失败: {str(e)}")
            return []
    
    def detect_white_player(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        识别玩家位置（白色三角）
        
        @param img: 输入图像（BGR格式）
        @return: 玩家位置坐标
        """
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)
            
            kernel = np.ones((5, 5), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
            
            return None
            
        except Exception as e:
            logger.error(f"识别玩家位置失败: {str(e)}")
            return None
    
    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两点之间的距离"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx * dx + dy * dy)
    
    def find_nearest_enemy(self, player_pos: Tuple[int, int], 
                          enemy_positions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """找到最近的敌人"""
        if not enemy_positions:
            return None
        
        min_distance = float('inf')
        nearest_enemy = None
        
        for enemy_pos in enemy_positions:
            distance = self.calculate_distance(player_pos, enemy_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_enemy = enemy_pos
        
        return nearest_enemy
    
    def human_like_mouse_move(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> None:
        """模拟人类鼠标移动"""
        try:
            distance = self.calculate_distance(start_pos, end_pos)
            base_speed = random.uniform(300, 800)
            move_duration = distance / base_speed
            move_duration = max(self.MIN_MOVE_DURATION, min(move_duration, self.MAX_MOVE_DURATION))
            move_duration *= random.uniform(0.9, 1.1)
            
            step_interval = random.uniform(0.02, 0.03)
            steps = max(15, int(move_duration / step_interval))
            actual_step_interval = move_duration / steps
            
            for i in range(steps + 1):
                t = i / steps
                if t < 0.5:
                    eased_t = 4 * t * t * t
                else:
                    eased_t = 1 - pow(-2 * t + 2, 3) / 2
                
                if 0.2 < t < 0.8:
                    noise_x = random.uniform(-1.5, 1.5)
                    noise_y = random.uniform(-1.5, 1.5)
                else:
                    noise_x = random.uniform(-0.5, 0.5)
                    noise_y = random.uniform(-0.5, 0.5)
                
                current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * eased_t + noise_x
                current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * eased_t + noise_y
                
                current_x = max(0, min(current_x, self.SCREEN_WIDTH - 1))
                current_y = max(0, min(current_y, self.SCREEN_HEIGHT - 1))
                
                pyautogui.moveTo(int(current_x), int(current_y), duration=actual_step_interval)
                
        except Exception as e:
            logger.error(f"鼠标移动失败: {str(e)}")
            raise
    
    def click_join_battle(self) -> bool:
        """
        点击"加入战斗"按钮
        
        @return: 操作是否成功
        """
        try:
            logger.info("点击加入战斗按钮")
            # "加入战斗"按钮通常在屏幕中央上方，坐标大约 (960, 150)
            # 使用图像识别或固定坐标
            button_x = 960
            button_y = 150
            
            current_pos = pyautogui.position()
            self.human_like_mouse_move((current_pos.x, current_pos.y), (button_x, button_y))
            self.random_delay(0.1, 0.2)
            pyautogui.click(button='left')
            self.random_delay(0.5, 1.0)
            logger.info("已点击加入战斗按钮")
            return True
            
        except Exception as e:
            logger.error(f"点击加入战斗按钮失败: {str(e)}")
            return False
    
    def select_ship(self, ship_index: int = 0) -> bool:
        """
        选择战船（从底部战船选择栏）
        
        @param ship_index: 战船索引（0为第一个）
        @return: 操作是否成功
        """
        try:
            logger.info(f"选择第 {ship_index + 1} 艘战船")
            # 战船选择栏在屏幕底部，第一个战船大约在 (200, 1000)
            ship_x = 200 + ship_index * 150  # 每个战船间隔约150像素
            ship_y = 1000
            
            current_pos = pyautogui.position()
            self.human_like_mouse_move((current_pos.x, current_pos.y), (ship_x, ship_y))
            self.random_delay(0.1, 0.2)
            pyautogui.click(button='left')
            self.random_delay(0.5, 0.8)
            logger.info("已选择战船")
            return True
            
        except Exception as e:
            logger.error(f"选择战船失败: {str(e)}")
            return False
    
    def auto_navigate_to_enemy(self, enemy_pos: Tuple[int, int]) -> bool:
        """
        自动导航到敌人位置
        
        @param enemy_pos: 敌人位置
        @return: 操作是否成功
        """
        try:
            logger.info(f"自动导航到敌人位置: {enemy_pos}")
            
            # 按M键打开地图
            pyautogui.press('m')
            self.random_delay(1.0, 1.5)
            
            # 在地图上点击敌人附近位置
            click_x = enemy_pos[0] + random.randint(-20, 20)
            click_y = enemy_pos[1] + random.randint(-20, 20) - 50  # 在敌人前方一点
            
            click_x = max(0, min(click_x, self.SCREEN_WIDTH - 1))
            click_y = max(0, min(click_y, self.SCREEN_HEIGHT - 1))
            
            current_pos = pyautogui.position()
            self.human_like_mouse_move((current_pos.x, current_pos.y), (click_x, click_y))
            self.random_delay(0.1, 0.2)
            pyautogui.click(button='left')
            
            # 关闭地图
            pyautogui.press('m')
            self.random_delay(0.3, 0.5)
            
            logger.info("自动导航完成")
            return True
            
        except Exception as e:
            logger.error(f"自动导航失败: {str(e)}")
            return False
    
    def auto_aim_at_enemy(self, enemy_pos: Tuple[int, int]) -> bool:
        """
        自动瞄准敌人
        
        @param enemy_pos: 敌人位置（屏幕坐标）
        @return: 操作是否成功
        """
        try:
            # 将敌人位置移动到屏幕中央（瞄准位置）
            center_x = self.SCREEN_WIDTH // 2
            center_y = self.SCREEN_HEIGHT // 2
            
            # 计算需要移动的鼠标距离
            move_x = enemy_pos[0] - center_x
            move_y = enemy_pos[1] - center_y
            
            # 移动鼠标来瞄准
            current_pos = pyautogui.position()
            target_x = current_pos.x + move_x
            target_y = current_pos.y + move_y
            
            target_x = max(0, min(target_x, self.SCREEN_WIDTH - 1))
            target_y = max(0, min(target_y, self.SCREEN_HEIGHT - 1))
            
            self.human_like_mouse_move((current_pos.x, current_pos.y), (target_x, target_y))
            self.random_delay(0.1, 0.2)
            
            logger.debug(f"已瞄准敌人位置: {enemy_pos}")
            return True
            
        except Exception as e:
            logger.error(f"自动瞄准失败: {str(e)}")
            return False
    
    def auto_shoot(self) -> bool:
        """
        自动射击
        
        @return: 操作是否成功
        """
        try:
            current_time = time.time()
            if current_time - self.last_shot_time < self.SHOOT_COOLDOWN:
                return False  # 还在冷却中
            
            logger.info("执行自动射击")
            # 左键点击射击
            pyautogui.click(button='left')
            self.last_shot_time = current_time
            self.random_delay(0.2, 0.3)
            logger.info("射击完成")
            return True
            
        except Exception as e:
            logger.error(f"自动射击失败: {str(e)}")
            return False
    
    def check_ship_destroyed(self, img: np.ndarray) -> bool:
        """
        检查战舰是否被击毁
        
        @param img: 输入图像
        @return: 是否被击毁
        """
        try:
            # 检查底部中央区域是否有"您的战舰已被击毁"相关文字
            # 或者检查玩家血量是否为0
            bottom_region = img[900:1080, 600:1320]
            gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(binary)
            
            # 如果底部有大量白色文字，可能是被击毁提示
            if white_pixels > 10000:
                logger.info("检测到战舰被击毁")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查战舰状态失败: {str(e)}")
            return False
    
    def leave_battle(self) -> bool:
        """
        离开战斗
        
        @return: 操作是否成功
        """
        try:
            logger.info("离开战斗")
            # 按ESC键
            pyautogui.press('esc')
            self.random_delay(0.5, 0.8)
            
            # 如果有确认对话框，点击"是"或"离开战斗"
            # "是"按钮通常在屏幕中央偏左，坐标大约 (700, 600)
            yes_button_x = 700
            yes_button_y = 600
            
            current_pos = pyautogui.position()
            self.human_like_mouse_move((current_pos.x, current_pos.y), (yes_button_x, yes_button_y))
            self.random_delay(0.1, 0.2)
            pyautogui.click(button='left')
            self.random_delay(1.0, 2.0)
            
            logger.info("已离开战斗")
            return True
            
        except Exception as e:
            logger.error(f"离开战斗失败: {str(e)}")
            return False
    
    def wait_for_port(self, max_wait: int = 30) -> bool:
        """
        等待返回港口界面
        
        @param max_wait: 最大等待时间（秒）
        @return: 是否成功返回港口
        """
        try:
            logger.info("等待返回港口界面")
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                img = self.capture_screen()
                state = self.detect_game_state(img)
                
                if state == GameState.PORT:
                    logger.info("已返回港口界面")
                    return True
                
                self.random_delay(1.0, 2.0)
            
            logger.warning("等待返回港口超时")
            return False
            
        except Exception as e:
            logger.error(f"等待返回港口失败: {str(e)}")
            return False
    
    def battle_loop(self) -> bool:
        """
        单局战斗循环
        
        @return: 战斗是否成功完成
        """
        try:
            logger.info("=" * 50)
            logger.info("开始战斗循环")
            logger.info("=" * 50)
            
            battle_start_time = time.time()
            last_enemy_search = 0.0
            
            while self.running:
                img = self.capture_screen()
                state = self.detect_game_state(img)
                self.current_state = state
                
                if state == GameState.DESTROYED:
                    logger.info("战舰被击毁，准备离开战斗")
                    self.leave_battle()
                    self.wait_for_port()
                    return True
                
                elif state == GameState.IN_BATTLE:
                    current_time = time.time()
                    
                    # 定期搜索敌人
                    if current_time - last_enemy_search > self.ENEMY_SEARCH_INTERVAL:
                        logger.info("搜索敌人")
                        self.enemy_positions = self.detect_red_enemies(img)
                        self.player_pos = self.detect_white_player(img)
                        
                        if self.player_pos and self.enemy_positions:
                            self.target_enemy = self.find_nearest_enemy(
                                self.player_pos, self.enemy_positions)
                            
                            if self.target_enemy:
                                logger.info(f"找到目标敌人: {self.target_enemy}")
                                # 自动导航
                                self.auto_navigate_to_enemy(self.target_enemy)
                                # 自动瞄准
                                self.auto_aim_at_enemy(self.target_enemy)
                                # 自动射击
                                self.auto_shoot()
                        
                        last_enemy_search = current_time
                    
                    # 检查是否被击毁
                    if self.check_ship_destroyed(img):
                        logger.info("检测到战舰被击毁")
                        self.leave_battle()
                        self.wait_for_port()
                        return True
                    
                    self.random_delay(0.5, 1.0)
                
                elif state == GameState.PORT:
                    logger.info("意外返回港口界面，战斗结束")
                    return True
                
                else:
                    self.random_delay(1.0, 2.0)
                
                # 检查战斗时间（防止无限循环）
                if time.time() - battle_start_time > 1800:  # 30分钟超时
                    logger.warning("战斗超时，强制退出")
                    self.leave_battle()
                    self.wait_for_port()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"战斗循环失败: {str(e)}")
            return False
    
    def run(self) -> None:
        """
        主运行循环
        
        @raise Exception: 当运行失败时抛出异常
        """
        try:
            logger.info("=" * 50)
            logger.info("战舰世界自动游戏机器人启动")
            logger.info("=" * 50)
            
            while self.running:
                # 检测当前状态
                img = self.capture_screen()
                state = self.detect_game_state(img)
                self.current_state = state
                
                if state == GameState.PORT:
                    logger.info(f"第 {self.battle_count + 1} 局游戏")
                    
                    # 选择战船
                    self.select_ship(0)  # 选择第一艘战船
                    self.random_delay(1.0, 2.0)
                    
                    # 点击加入战斗
                    if self.click_join_battle():
                        self.battle_count += 1
                        self.random_delay(5.0, 8.0)  # 等待加载
                        
                        # 进入战斗循环
                        self.battle_loop()
                        
                        # 等待返回港口
                        self.wait_for_port(30)
                        self.random_delay(3.0, 5.0)  # 等待结算完成
                
                elif state == GameState.IN_BATTLE:
                    logger.info("已在战斗中，直接进入战斗循环")
                    self.battle_loop()
                    self.wait_for_port(30)
                
                else:
                    logger.warning(f"未知状态: {state}，等待...")
                    self.random_delay(2.0, 3.0)
            
        except KeyboardInterrupt:
            logger.info("用户中断，退出程序")
            self.running = False
        except Exception as e:
            logger.error(f"运行失败: {str(e)}", exc_info=True)
            raise


def main():
    """主函数"""
    try:
        logger.info("程序启动")
        bot = WowsAutoBot()
        bot.run()
        logger.info("程序结束")
        return 0
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

