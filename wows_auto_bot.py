"""
战舰世界自动游戏脚本
功能：自动选择战船、开始游戏、寻敌、瞄准、射击、检测被击毁、退出并开始下一局

@author seven
@since 2025-12-23
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
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wows_auto_bot.log', encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 配置pyautogui安全设置
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# 创建调试输出目录
DEBUG_DIR = Path("debug_output")
DEBUG_DIR.mkdir(exist_ok=True)


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
    """战舰世界自动游戏机器人
    
    该机器人可以自动完成战舰世界游戏的完整流程，包括：
    - 识别游戏状态
    - 点击加入战斗按钮
    - 自动寻敌、导航、瞄准和射击
    - 检测被击毁并返回港口
    """
    
    # 屏幕分辨率
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    
    # 颜色范围定义（HSV颜色空间）
    RED_LOWER_1 = np.array([0, 100, 100])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([170, 100, 100])
    RED_UPPER_2 = np.array([180, 255, 255])
    
    # 橙色范围（加入战斗按钮）
    ORANGE_LOWER = np.array([10, 100, 100])
    ORANGE_UPPER = np.array([25, 255, 255])
    
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
    
    # 按钮位置（基于1920x1080分辨率）
    JOIN_BATTLE_BUTTON_REGION = (650, 0, 850, 50)  # 加入战斗按钮区域 (x1, y1, x2, y2)
    
    def __init__(self, debug_mode: bool = False):
        """
        初始化自动游戏机器人
        
        @param debug_mode: 是否启用调试模式（保存截图和详细日志）
        @since 2025-12-23
        """
        logger.info("=" * 60)
        logger.info("初始化战舰世界自动游戏机器人")
        logger.info(f"屏幕分辨率: {self.SCREEN_WIDTH}x{self.SCREEN_HEIGHT}")
        logger.info(f"调试模式: {'开启' if debug_mode else '关闭'}")
        logger.info("=" * 60)
        
        self.debug_mode = debug_mode
        self.current_state = GameState.UNKNOWN
        self.player_pos: Optional[Tuple[int, int]] = None
        self.enemy_positions: List[Tuple[int, int]] = []
        self.target_enemy: Optional[Tuple[int, int]] = None
        self.last_shot_time = 0.0
        self.battle_count = 0
        self.running = True
        self.screenshot_counter = 0
        
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
    
    def save_debug_screenshot(self, img: np.ndarray, prefix: str = "screenshot") -> None:
        """
        保存调试截图
        
        @param img: 要保存的图像
        @param prefix: 文件名前缀
        @since 2025-12-23
        """
        if not self.debug_mode:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = DEBUG_DIR / f"{prefix}_{timestamp}_{self.screenshot_counter}.png"
            cv2.imwrite(str(filename), img)
            self.screenshot_counter += 1
            logger.debug(f"保存调试截图: {filename}")
        except Exception as e:
            logger.warning(f"保存调试截图失败: {str(e)}")
    
    def capture_screen(self) -> np.ndarray:
        """
        捕获当前屏幕截图
        
        @return: 屏幕截图的numpy数组（BGR格式）
        @raise Exception: 当截图失败时抛出异常
        @since 2025-12-23
        """
        try:
            logger.debug("捕获屏幕截图")
            screenshot = pyautogui.screenshot()
            img_array = np.array(screenshot)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            logger.debug(f"截图大小: {img_bgr.shape}")
            return img_bgr
        except Exception as e:
            logger.error(f"捕获屏幕截图失败: {str(e)}", exc_info=True)
            raise
    
    def detect_orange_button(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        检测橙色"加入战斗"按钮
        
        @param img: 输入图像（BGR格式）
        @return: 按钮中心坐标，如果未找到则返回None
        @since 2025-12-23
        """
        try:
            # 提取按钮区域
            x1, y1, x2, y2 = self.JOIN_BATTLE_BUTTON_REGION
            button_region = img[y1:y2, x1:x2]
            
            # 转换为HSV并检测橙色
            hsv = cv2.cvtColor(button_region, cv2.COLOR_BGR2HSV)
            orange_mask = cv2.inRange(hsv, self.ORANGE_LOWER, self.ORANGE_UPPER)
            orange_pixels = cv2.countNonZero(orange_mask)
            
            logger.debug(f"橙色按钮区域检测到橙色像素数: {orange_pixels}")
            
            # 如果橙色像素数量足够，认为找到按钮
            if orange_pixels > 100:
                # 返回按钮区域的中心坐标（转换为全屏坐标）
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2
                logger.info(f"检测到橙色加入战斗按钮，位置: ({center_x}, {center_y})")
                
                if self.debug_mode:
                    debug_img = img.copy()
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(debug_img, (center_x, center_y), 5, (0, 0, 255), -1)
                    self.save_debug_screenshot(debug_img, "orange_button_detected")
                
                return (center_x, center_y)
            
            return None
            
        except Exception as e:
            logger.error(f"检测橙色按钮失败: {str(e)}", exc_info=True)
            return None
    
    def detect_game_state(self, img: np.ndarray) -> GameState:
        """
        检测当前游戏状态
        
        使用多种检测方法综合判断当前游戏状态：
        1. 检测橙色"加入战斗"按钮 -> PORT
        2. 检测底部战船选择栏 -> PORT
        3. 检测被击毁提示文字 -> DESTROYED
        4. 检测小地图（右下角） -> IN_BATTLE
        5. 检测全屏地图网格 -> MAP_OPEN
        
        @param img: 输入图像（BGR格式）
        @return: 游戏状态
        @since 2025-12-23
        """
        try:
            logger.debug("开始检测游戏状态")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 1. 检测橙色"加入战斗"按钮（最可靠的港口界面标志）
            orange_button = self.detect_orange_button(img)
            if orange_button is not None:
                logger.info("检测到港口界面（橙色加入战斗按钮）")
                self.save_debug_screenshot(img, "state_port")
                return GameState.PORT
            
            # 2. 检测底部战船选择栏（港口界面特征）
            # 战船选择栏通常在屏幕底部，有深色背景和战船图标
            bottom_ship_region = img[950:1080, 0:1920]
            bottom_gray = cv2.cvtColor(bottom_ship_region, cv2.COLOR_BGR2GRAY)
            # 检查底部区域的平均亮度（战船选择栏通常较暗）
            mean_brightness = np.mean(bottom_gray)
            logger.debug(f"底部区域平均亮度: {mean_brightness}")
            
            # 检测底部是否有多个船只图标（检测明暗对比）
            _, bottom_binary = cv2.threshold(bottom_gray, 50, 255, cv2.THRESH_BINARY)
            white_pixels_bottom = cv2.countNonZero(bottom_binary)
            logger.debug(f"底部区域白色像素数: {white_pixels_bottom}")
            
            if mean_brightness < 80 and white_pixels_bottom > 5000:
                # 底部较暗且有适量明亮区域，可能是港口界面
                logger.info("检测到港口界面（底部战船选择栏特征）")
                self.save_debug_screenshot(img, "state_port_ships")
                return GameState.PORT
            
            # 3. 检测被击毁界面（底部中央有大量白色文字）
            destroyed_region = img[900:1000, 600:1320]
            destroyed_gray = cv2.cvtColor(destroyed_region, cv2.COLOR_BGR2GRAY)
            _, destroyed_binary = cv2.threshold(destroyed_gray, 200, 255, cv2.THRESH_BINARY)
            white_pixels_destroyed = cv2.countNonZero(destroyed_binary)
            logger.debug(f"被击毁区域白色像素数: {white_pixels_destroyed}")
            
            if white_pixels_destroyed > 8000:
                logger.info("检测到被击毁界面")
                self.save_debug_screenshot(img, "state_destroyed")
                return GameState.DESTROYED
            
            # 4. 检测右下角小地图（战斗中的标志）
            # 小地图通常在右下角，有深蓝色背景
            minimap_region = img[400:750, 1100:1450]
            minimap_hsv = cv2.cvtColor(minimap_region, cv2.COLOR_BGR2HSV)
            
            # 检测深蓝色（小地图背景）
            blue_lower = np.array([100, 50, 30])
            blue_upper = np.array([130, 255, 150])
            blue_mask = cv2.inRange(minimap_hsv, blue_lower, blue_upper)
            blue_pixels = cv2.countNonZero(blue_mask)
            logger.debug(f"小地图区域蓝色像素数: {blue_pixels}")
            
            if blue_pixels > 5000:
                logger.info("检测到战斗界面（小地图）")
                self.save_debug_screenshot(img, "state_battle")
                return GameState.IN_BATTLE
            
            # 5. 检测全屏地图（按M键打开的大地图）
            # 全屏地图有大量网格线
            map_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(map_gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 30:
                logger.info("检测到全屏地图界面")
                self.save_debug_screenshot(img, "state_map_open")
                return GameState.MAP_OPEN
            
            # 无法确定状态
            logger.warning("无法确定游戏状态，返回未知状态")
            self.save_debug_screenshot(img, "state_unknown")
            return GameState.UNKNOWN
            
        except Exception as e:
            logger.error(f"检测游戏状态失败: {str(e)}", exc_info=True)
            self.save_debug_screenshot(img, "state_error")
            return GameState.UNKNOWN
    
    def detect_red_enemies(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """
        识别图像中的红色多边形敌人标记
        
        在小地图上检测红色敌方船只标记。敌人标记通常是红色的多边形图标。
        
        @param img: 输入图像（BGR格式）
        @return: 敌人位置列表（屏幕坐标）
        @since 2025-12-23
        """
        try:
            logger.debug("开始识别红色敌人")
            
            # 只在右下角小地图区域检测（提高效率和准确性）
            minimap_region = img[400:750, 1100:1450]
            minimap_offset_x = 1100
            minimap_offset_y = 400
            
            hsv = cv2.cvtColor(minimap_region, cv2.COLOR_BGR2HSV)
            
            # 检测红色（两个范围）
            mask1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
            mask2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # 形态学处理，去除噪点
            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            enemy_positions = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # 过滤过小的轮廓
                if area < 30:
                    continue
                
                # 近似多边形
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = len(approx)
                
                # 过滤不合理的形状
                if vertices < 3 or vertices > 6:
                    continue
                
                # 计算质心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + minimap_offset_x
                    cy = int(M["m01"] / M["m00"]) + minimap_offset_y
                    
                    # 计算实心度（排除不规则形状）
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity > 0.7:
                            enemy_positions.append((cx, cy))
                            logger.debug(f"找到敌人 #{i+1}: 位置=({cx}, {cy}), 面积={area:.1f}, 实心度={solidity:.2f}")
            
            logger.info(f"共识别到 {len(enemy_positions)} 个敌人")
            
            # 保存调试图
            if self.debug_mode and len(enemy_positions) > 0:
                debug_img = img.copy()
                for pos in enemy_positions:
                    cv2.circle(debug_img, pos, 10, (0, 255, 0), 2)
                self.save_debug_screenshot(debug_img, "enemies_detected")
            
            return enemy_positions
            
        except Exception as e:
            logger.error(f"识别红色敌人失败 - 异常: {str(e)}", exc_info=True)
            return []
    
    def detect_white_player(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        识别玩家位置（白色三角标记）
        
        在小地图上检测代表玩家自己的白色三角形标记。
        
        @param img: 输入图像（BGR格式）
        @return: 玩家位置坐标（屏幕坐标），未找到返回None
        @since 2025-12-23
        """
        try:
            logger.debug("开始识别玩家位置")
            
            # 只在右下角小地图区域检测
            minimap_region = img[400:750, 1100:1450]
            minimap_offset_x = 1100
            minimap_offset_y = 400
            
            hsv = cv2.cvtColor(minimap_region, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)
            
            # 形态学处理
            kernel = np.ones((5, 5), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.debug("未找到玩家位置标记")
                return None
            
            # 找到最大的轮廓（玩家标记通常是最明显的白色区域）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 计算质心
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + minimap_offset_x
                cy = int(M["m01"] / M["m00"]) + minimap_offset_y
                logger.info(f"识别到玩家位置: ({cx}, {cy}), 标记面积: {area:.1f}")
                
                # 保存调试图
                if self.debug_mode:
                    debug_img = img.copy()
                    cv2.circle(debug_img, (cx, cy), 15, (255, 0, 0), 3)
                    self.save_debug_screenshot(debug_img, "player_detected")
                
                return (cx, cy)
            
            logger.debug("玩家标记质心计算失败")
            return None
            
        except Exception as e:
            logger.error(f"识别玩家位置失败 - 异常: {str(e)}", exc_info=True)
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
        
        首先尝试检测橙色按钮的实际位置，如果检测失败则使用默认坐标。
        
        @return: 操作是否成功
        @since 2025-12-23
        """
        try:
            logger.info("=" * 50)
            logger.info("准备点击加入战斗按钮")
            
            # 先截图检测按钮位置
            img = self.capture_screen()
            button_pos = self.detect_orange_button(img)
            
            if button_pos is not None:
                button_x, button_y = button_pos
                logger.info(f"使用检测到的按钮位置: ({button_x}, {button_y})")
            else:
                # 使用默认位置（右上角橙色区域的中心）
                button_x = 730
                button_y = 25
                logger.warning(f"未能检测到按钮，使用默认位置: ({button_x}, {button_y})")
            
            # 移动鼠标到按钮位置
            current_pos = pyautogui.position()
            logger.info(f"鼠标从 ({current_pos.x}, {current_pos.y}) 移动到 ({button_x}, {button_y})")
            
            self.human_like_mouse_move((current_pos.x, current_pos.y), (button_x, button_y))
            self.random_delay(0.2, 0.4)
            
            # 点击按钮
            pyautogui.click(button='left')
            logger.info("已点击加入战斗按钮")
            
            # 等待加载
            self.random_delay(0.5, 1.0)
            
            # 保存点击后的截图用于调试
            if self.debug_mode:
                after_click_img = self.capture_screen()
                self.save_debug_screenshot(after_click_img, "after_join_battle_click")
            
            logger.info("=" * 50)
            return True
            
        except Exception as e:
            logger.error(f"点击加入战斗按钮失败 - URL: N/A, 参数: button_pos={button_pos if 'button_pos' in locals() else 'None'}, 异常: {str(e)}", exc_info=True)
            return False
    
    def select_ship(self, ship_index: int = 0) -> bool:
        """
        选择战船（从底部战船选择栏）
        
        @param ship_index: 战船索引（0为第一个）
        @return: 操作是否成功
        @since 2025-12-23
        """
        try:
            logger.info(f"选择第 {ship_index + 1} 艘战船")
            
            # 战船选择栏在屏幕底部，第一个战船大约在 (100, 1000)
            # 根据港口界面图片，战船图标间隔约150-180像素
            ship_x = 100 + ship_index * 165
            ship_y = 1000
            
            logger.info(f"战船位置: ({ship_x}, {ship_y})")
            
            current_pos = pyautogui.position()
            self.human_like_mouse_move((current_pos.x, current_pos.y), (ship_x, ship_y))
            self.random_delay(0.2, 0.3)
            
            pyautogui.click(button='left')
            logger.info(f"已点击战船 #{ship_index + 1}")
            
            self.random_delay(0.5, 0.8)
            
            if self.debug_mode:
                after_select_img = self.capture_screen()
                self.save_debug_screenshot(after_select_img, f"after_select_ship_{ship_index}")
            
            return True
            
        except Exception as e:
            logger.error(f"选择战船失败 - 参数: ship_index={ship_index}, 位置=({ship_x if 'ship_x' in locals() else 'N/A'}, {ship_y if 'ship_y' in locals() else 'N/A'}), 异常: {str(e)}", exc_info=True)
            return False
    
    def auto_navigate_to_enemy(self, enemy_pos: Tuple[int, int]) -> bool:
        """
        自动导航到敌人位置
        
        通过打开地图并在敌人位置附近点击来设置自动导航目标。
        
        @param enemy_pos: 敌人位置（小地图坐标）
        @return: 操作是否成功
        @since 2025-12-23
        """
        try:
            logger.info(f"开始自动导航到敌人位置: {enemy_pos}")
            
            # 按M键打开大地图
            logger.debug("按M键打开大地图")
            pyautogui.press('m')
            self.random_delay(1.0, 1.5)
            
            # 在地图上点击敌人附近位置（略微前方，以便接近敌人）
            offset_x = random.randint(-20, 20)
            offset_y = random.randint(-60, -40)  # 在敌人前方
            click_x = enemy_pos[0] + offset_x
            click_y = enemy_pos[1] + offset_y
            
            # 确保点击位置在屏幕范围内
            click_x = max(50, min(click_x, self.SCREEN_WIDTH - 50))
            click_y = max(50, min(click_y, self.SCREEN_HEIGHT - 50))
            
            logger.info(f"在地图上点击导航目标: ({click_x}, {click_y})")
            
            current_pos = pyautogui.position()
            self.human_like_mouse_move((current_pos.x, current_pos.y), (click_x, click_y))
            self.random_delay(0.2, 0.3)
            pyautogui.click(button='left')
            self.random_delay(0.3, 0.5)
            
            # 关闭地图
            logger.debug("按M键关闭大地图")
            pyautogui.press('m')
            self.random_delay(0.3, 0.5)
            
            logger.info("自动导航设置完成")
            return True
            
        except Exception as e:
            logger.error(f"自动导航失败 - 目标位置: {enemy_pos}, 异常: {str(e)}", exc_info=True)
            # 尝试关闭地图
            try:
                pyautogui.press('m')
            except:
                pass
            return False
    
    def auto_aim_at_enemy(self, enemy_pos: Tuple[int, int]) -> bool:
        """
        自动瞄准敌人
        
        通过移动鼠标将敌人移动到屏幕中央（瞄准位置）。
        注意：这个方法假设摄像机会随鼠标移动而旋转。
        
        @param enemy_pos: 敌人位置（小地图坐标）
        @return: 操作是否成功
        @since 2025-12-23
        """
        try:
            logger.debug(f"开始瞄准敌人，目标位置: {enemy_pos}")
            
            # 屏幕中心（瞄准点）
            center_x = self.SCREEN_WIDTH // 2
            center_y = self.SCREEN_HEIGHT // 2
            
            # 计算小地图位置到屏幕中心的偏移
            # 注意：这里的逻辑可能需要根据实际游戏调整
            # 小地图坐标不能直接用于3D瞄准，这里只是简单演示
            move_x = (enemy_pos[0] - 1275) * 2  # 1275是小地图中心X
            move_y = (enemy_pos[1] - 575) * 2   # 575是小地图中心Y
            
            # 移动鼠标来旋转视角
            current_pos = pyautogui.position()
            target_x = current_pos.x + move_x
            target_y = current_pos.y + move_y
            
            # 限制在屏幕范围内
            target_x = max(50, min(target_x, self.SCREEN_WIDTH - 50))
            target_y = max(50, min(target_y, self.SCREEN_HEIGHT - 50))
            
            logger.debug(f"移动鼠标: 从({current_pos.x}, {current_pos.y})到({target_x}, {target_y})")
            self.human_like_mouse_move((current_pos.x, current_pos.y), (target_x, target_y))
            self.random_delay(0.2, 0.3)
            
            logger.debug("瞄准完成")
            return True
            
        except Exception as e:
            logger.error(f"自动瞄准失败 - 目标: {enemy_pos}, 异常: {str(e)}", exc_info=True)
            return False
    
    def auto_shoot(self) -> bool:
        """
        自动射击
        
        检查射击冷却时间，如果冷却完毕则执行射击。
        
        @return: 操作是否成功
        @since 2025-12-23
        """
        try:
            current_time = time.time()
            time_since_last_shot = current_time - self.last_shot_time
            
            # 检查冷却时间
            if time_since_last_shot < self.SHOOT_COOLDOWN:
                logger.debug(f"射击冷却中，剩余: {self.SHOOT_COOLDOWN - time_since_last_shot:.1f}秒")
                return False
            
            logger.info("执行自动射击")
            
            # 左键点击射击
            pyautogui.click(button='left')
            self.last_shot_time = current_time
            
            self.random_delay(0.2, 0.3)
            logger.info("射击完成")
            return True
            
        except Exception as e:
            logger.error(f"自动射击失败 - 异常: {str(e)}", exc_info=True)
            return False
    
    def check_ship_destroyed(self, img: np.ndarray) -> bool:
        """
        检查战舰是否被击毁
        
        通过检测底部中央区域的白色文字来判断是否被击毁。
        被击毁时通常会显示"您的战舰已被击毁"等提示文字。
        
        @param img: 输入图像（BGR格式）
        @return: 是否被击毁
        @since 2025-12-23
        """
        try:
            # 检查底部中央区域是否有"您的战舰已被击毁"相关文字
            bottom_region = img[900:1000, 600:1320]
            gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
            
            # 二值化，提取白色文字
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(binary)
            
            logger.debug(f"被击毁检测区域白色像素数: {white_pixels}")
            
            # 如果底部有大量白色文字，可能是被击毁提示
            if white_pixels > 8000:
                logger.warning("检测到战舰被击毁！")
                
                if self.debug_mode:
                    self.save_debug_screenshot(img, "ship_destroyed")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查战舰状态失败 - 异常: {str(e)}", exc_info=True)
            return False
    
    def leave_battle(self) -> bool:
        """
        离开战斗
        
        按ESC键打开菜单，然后点击确认按钮离开战斗。
        
        @return: 操作是否成功
        @since 2025-12-23
        """
        try:
            logger.info("=" * 50)
            logger.info("准备离开战斗")
            
            # 按ESC键打开菜单
            logger.debug("按ESC键打开菜单")
            pyautogui.press('esc')
            self.random_delay(0.8, 1.2)
            
            # 点击"离开战斗"或"确认"按钮
            # 按钮通常在屏幕中央偏下，坐标大约 (960, 600)
            yes_button_x = 960
            yes_button_y = 600
            
            logger.debug(f"点击确认按钮: ({yes_button_x}, {yes_button_y})")
            
            current_pos = pyautogui.position()
            self.human_like_mouse_move((current_pos.x, current_pos.y), (yes_button_x, yes_button_y))
            self.random_delay(0.2, 0.3)
            pyautogui.click(button='left')
            
            self.random_delay(1.0, 2.0)
            
            logger.info("已发送离开战斗指令")
            logger.info("=" * 50)
            return True
            
        except Exception as e:
            logger.error(f"离开战斗失败 - 异常: {str(e)}", exc_info=True)
            return False
    
    def wait_for_port(self, max_wait: int = 30) -> bool:
        """
        等待返回港口界面
        
        持续检测游戏状态，直到检测到港口界面或超时。
        
        @param max_wait: 最大等待时间（秒）
        @return: 是否成功返回港口
        @since 2025-12-23
        """
        try:
            logger.info("=" * 50)
            logger.info(f"等待返回港口界面（最长等待{max_wait}秒）")
            start_time = time.time()
            check_count = 0
            
            while time.time() - start_time < max_wait:
                check_count += 1
                elapsed = time.time() - start_time
                
                logger.debug(f"检查 #{check_count}, 已等待 {elapsed:.1f}秒")
                
                img = self.capture_screen()
                state = self.detect_game_state(img)
                
                if state == GameState.PORT:
                    logger.info(f"已返回港口界面（耗时 {elapsed:.1f}秒）")
                    logger.info("=" * 50)
                    return True
                
                logger.debug(f"当前状态: {state}, 继续等待...")
                self.random_delay(2.0, 3.0)
            
            logger.warning(f"等待返回港口超时（{max_wait}秒）")
            logger.info("=" * 50)
            return False
            
        except Exception as e:
            logger.error(f"等待返回港口失败 - 异常: {str(e)}", exc_info=True)
            logger.info("=" * 50)
            return False
    
    def battle_loop(self) -> bool:
        """
        单局战斗循环
        
        在战斗中持续执行以下操作：
        1. 检测游戏状态
        2. 搜索敌人
        3. 导航到最近的敌人
        4. 瞄准并射击
        5. 检测是否被击毁
        
        @return: 战斗是否成功完成
        @since 2025-12-23
        """
        try:
            logger.info("=" * 60)
            logger.info("开始战斗循环")
            logger.info("=" * 60)
            
            battle_start_time = time.time()
            last_enemy_search = 0.0
            loop_count = 0
            
            while self.running:
                loop_count += 1
                elapsed = time.time() - battle_start_time
                
                logger.debug(f"战斗循环 #{loop_count}, 已进行 {elapsed:.1f}秒")
                
                # 捕获屏幕并检测状态
                img = self.capture_screen()
                state = self.detect_game_state(img)
                self.current_state = state
                
                # 状态1: 被击毁
                if state == GameState.DESTROYED:
                    logger.warning("战舰被击毁，准备离开战斗")
                    self.leave_battle()
                    self.wait_for_port(45)
                    return True
                
                # 状态2: 战斗中
                elif state == GameState.IN_BATTLE:
                    current_time = time.time()
                    
                    # 定期搜索敌人
                    if current_time - last_enemy_search > self.ENEMY_SEARCH_INTERVAL:
                        logger.info("-" * 40)
                        logger.info("执行寻敌和攻击流程")
                        
                        # 识别敌人和玩家位置
                        self.enemy_positions = self.detect_red_enemies(img)
                        self.player_pos = self.detect_white_player(img)
                        
                        if self.player_pos:
                            logger.info(f"玩家位置: {self.player_pos}")
                        else:
                            logger.warning("未能识别玩家位置")
                        
                        if self.enemy_positions:
                            logger.info(f"发现 {len(self.enemy_positions)} 个敌人")
                            
                            # 找到最近的敌人
                            if self.player_pos:
                                self.target_enemy = self.find_nearest_enemy(
                                    self.player_pos, self.enemy_positions)
                                
                                if self.target_enemy:
                                    distance = self.calculate_distance(
                                        self.player_pos, self.target_enemy)
                                    logger.info(f"锁定目标: {self.target_enemy}, 距离: {distance:.1f}")
                                    
                                    # 自动导航到敌人
                                    self.auto_navigate_to_enemy(self.target_enemy)
                                    
                                    # 自动瞄准
                                    self.auto_aim_at_enemy(self.target_enemy)
                                    
                                    # 自动射击
                                    self.auto_shoot()
                            else:
                                # 没有玩家位置，随便选一个敌人
                                self.target_enemy = self.enemy_positions[0]
                                logger.info(f"选择第一个敌人作为目标: {self.target_enemy}")
                                self.auto_navigate_to_enemy(self.target_enemy)
                        else:
                            logger.warning("未发现敌人")
                        
                        logger.info("-" * 40)
                        last_enemy_search = current_time
                    
                    # 额外检查是否被击毁
                    if self.check_ship_destroyed(img):
                        logger.warning("检测到战舰被击毁（额外检查）")
                        self.leave_battle()
                        self.wait_for_port(45)
                        return True
                    
                    self.random_delay(0.8, 1.2)
                
                # 状态3: 意外返回港口
                elif state == GameState.PORT:
                    logger.info("检测到已返回港口界面，战斗结束")
                    return True
                
                # 状态4: 地图打开（关闭它）
                elif state == GameState.MAP_OPEN:
                    logger.debug("检测到地图打开，关闭地图")
                    pyautogui.press('m')
                    self.random_delay(0.5, 0.8)
                
                # 状态5: 未知状态
                else:
                    logger.debug(f"未知状态: {state}, 继续等待...")
                    self.random_delay(1.5, 2.5)
                
                # 检查战斗时间（防止无限循环）
                if elapsed > 1800:  # 30分钟超时
                    logger.warning("战斗超时（30分钟），强制退出")
                    self.leave_battle()
                    self.wait_for_port(45)
                    return True
            
            logger.info("战斗循环被中断")
            return False
            
        except Exception as e:
            logger.error(f"战斗循环失败 - 异常: {str(e)}", exc_info=True)
            # 尝试离开战斗
            try:
                self.leave_battle()
                self.wait_for_port(45)
            except:
                pass
            return False
    
    def run(self) -> None:
        """
        主运行循环
        
        持续检测游戏状态并执行相应操作：
        - 在港口界面：选择战船并加入战斗
        - 在战斗中：执行战斗循环
        - 其他状态：等待并重试
        
        @raise Exception: 当运行失败时抛出异常
        @since 2025-12-23
        """
        try:
            logger.info("=" * 60)
            logger.info("战舰世界自动游戏机器人启动")
            logger.info(f"调试模式: {'开启' if self.debug_mode else '关闭'}")
            logger.info("按 Ctrl+C 停止程序")
            logger.info("=" * 60)
            
            main_loop_count = 0
            
            while self.running:
                main_loop_count += 1
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"主循环 #{main_loop_count}")
                logger.info("=" * 60)
                
                # 检测当前状态
                img = self.capture_screen()
                state = self.detect_game_state(img)
                self.current_state = state
                
                logger.info(f"当前游戏状态: {state.value}")
                
                # 处理不同的游戏状态
                if state == GameState.PORT:
                    # 在港口界面，开始新的一局
                    logger.info("")
                    logger.info("*" * 60)
                    logger.info(f"准备开始第 {self.battle_count + 1} 局游戏")
                    logger.info("*" * 60)
                    
                    # 选择战船
                    logger.info("步骤1: 选择战船")
                    self.select_ship(0)  # 选择第一艘战船
                    self.random_delay(1.5, 2.5)
                    
                    # 点击加入战斗
                    logger.info("步骤2: 点击加入战斗")
                    if self.click_join_battle():
                        self.battle_count += 1
                        logger.info(f"成功加入战斗，等待加载...")
                        self.random_delay(5.0, 8.0)  # 等待加载
                        
                        # 进入战斗循环
                        logger.info("步骤3: 进入战斗")
                        self.battle_loop()
                        
                        # 等待返回港口
                        logger.info("步骤4: 等待返回港口")
                        self.wait_for_port(60)
                        self.random_delay(3.0, 5.0)  # 等待结算完成
                    else:
                        logger.error("点击加入战斗失败，重试...")
                        self.random_delay(3.0, 5.0)
                
                elif state == GameState.IN_BATTLE:
                    # 已经在战斗中，直接进入战斗循环
                    logger.info("检测到已在战斗中，直接进入战斗循环")
                    self.battle_loop()
                    self.wait_for_port(60)
                
                elif state == GameState.DESTROYED:
                    # 被击毁，尝试离开战斗
                    logger.info("检测到被击毁状态，尝试离开战斗")
                    self.leave_battle()
                    self.wait_for_port(60)
                
                elif state == GameState.MAP_OPEN:
                    # 地图打开，关闭它
                    logger.info("检测到地图打开，关闭地图")
                    pyautogui.press('m')
                    self.random_delay(1.0, 1.5)
                
                else:
                    # 未知状态，等待
                    logger.warning(f"未知状态: {state.value}，等待后重试...")
                    self.random_delay(3.0, 5.0)
            
            logger.info("=" * 60)
            logger.info("程序正常退出")
            logger.info("=" * 60)
            
        except KeyboardInterrupt:
            logger.info("")
            logger.info("=" * 60)
            logger.info("用户中断（Ctrl+C），退出程序")
            logger.info("=" * 60)
            self.running = False
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"运行失败 - 异常: {str(e)}", exc_info=True)
            logger.error("=" * 60)
            raise


def main():
    """
    主函数
    
    支持命令行参数：
    - --debug: 启用调试模式，保存关键截图
    
    @return: 退出码（0表示成功，1表示失败）
    @since 2025-12-23
    """
    try:
        import argparse
        
        # 解析命令行参数
        parser = argparse.ArgumentParser(
            description='战舰世界自动游戏机器人',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
示例:
  python wows_auto_bot.py           # 正常模式运行
  python wows_auto_bot.py --debug   # 调试模式运行（保存截图）

作者: seven
时间: 2025-12-23
            '''
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='启用调试模式，保存关键截图到debug_output目录'
        )
        
        args = parser.parse_args()
        
        logger.info("")
        logger.info("*" * 60)
        logger.info("程序启动")
        logger.info(f"调试模式: {'开启' if args.debug else '关闭'}")
        logger.info("*" * 60)
        logger.info("")
        
        # 创建机器人实例并运行
        bot = WowsAutoBot(debug_mode=args.debug)
        bot.run()
        
        logger.info("")
        logger.info("*" * 60)
        logger.info("程序结束")
        logger.info(f"总共完成 {bot.battle_count} 局游戏")
        logger.info("*" * 60)
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error("*" * 60)
        logger.error(f"程序执行失败 - 异常: {str(e)}", exc_info=True)
        logger.error("*" * 60)
        logger.error("")
        return 1


if __name__ == "__main__":
    exit(main())

