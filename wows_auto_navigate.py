"""
战舰世界自动导航脚本
功能：识别离自己最近的敌人，打开地图，点击敌人附近区域进行自动导航

@author seven
@since 2024-12-19
"""

import cv2
import numpy as np
import pyautogui
import time
import logging
from typing import Tuple, List, Optional
import sys
import random
from datetime import datetime
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wows_auto_navigate.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 配置pyautogui安全设置
pyautogui.FAILSAFE = True  # 鼠标移到屏幕角落可以中断
pyautogui.PAUSE = 0.1  # 操作间隔


class WowsAutoNavigate:
    """战舰世界自动导航类"""
    
    # 颜色范围定义（HSV颜色空间）
    # 红色敌人（HSV范围）
    RED_LOWER_1 = np.array([0, 100, 100])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([170, 100, 100])
    RED_UPPER_2 = np.array([180, 255, 255])
    
    # 白色三角自己（HSV范围）
    WHITE_LOWER = np.array([0, 0, 200])
    WHITE_UPPER = np.array([180, 30, 255])
    
    # 屏幕分辨率
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    
    # 鼠标移动参数（模拟人类操作）
    MIN_MOVE_DURATION = 0.5  # 最小移动时间（秒）
    MAX_MOVE_DURATION = 1.5  # 最大移动时间（秒）
    MIN_RANDOM_DELAY = 0.1   # 最小随机延迟（秒）
    MAX_RANDOM_DELAY = 0.3   # 最大随机延迟（秒）
    
    def __init__(self):
        """初始化自动导航系统"""
        logger.info("初始化战舰世界自动导航系统")
        self.player_pos: Optional[Tuple[int, int]] = None
        self.enemy_positions: List[Tuple[int, int]] = []
        self.nearest_enemy: Optional[Tuple[int, int]] = None
        self.screenshot_img: Optional[np.ndarray] = None
        
    def capture_screen(self) -> np.ndarray:
        """
        捕获当前屏幕截图
        
        @return: 屏幕截图的numpy数组（BGR格式）
        @raise Exception: 当截图失败时抛出异常
        """
        try:
            logger.info("开始捕获屏幕截图")
            screenshot = pyautogui.screenshot()
            # 转换为numpy数组并转换为BGR格式（OpenCV使用）
            img_array = np.array(screenshot)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            logger.info(f"屏幕截图捕获成功，尺寸: {img_bgr.shape}")
            return img_bgr
        except Exception as e:
            logger.error(f"捕获屏幕截图失败: {str(e)}")
            raise
    
    def detect_red_enemies(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """
        识别图像中的红色多边形敌人（三角形标记）
        
        @param img: 输入图像（BGR格式）
        @return: 敌人位置列表，每个元素为(x, y)坐标
        @raise Exception: 当图像处理失败时抛出异常
        """
        try:
            logger.info("开始识别红色多边形敌人")
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 创建红色掩码（红色在HSV中跨越0和180度）
            mask1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
            mask2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # 形态学操作，去除噪声，保留多边形形状
            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            enemy_positions = []
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                
                # 过滤太小的区域（可能是噪声）
                if area < 30:  # 最小面积阈值
                    continue
                
                # 检测是否为多边形（三角形或接近三角形的形状）
                # 使用approxPolyDP来近似轮廓
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 检查顶点数量，三角形应该有3个顶点，但允许一定误差（3-5个顶点）
                vertices = len(approx)
                if vertices < 3 or vertices > 6:
                    # 不是多边形，可能是其他形状，跳过
                    continue
                
                # 计算轮廓的中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 计算凸包，进一步验证是否为三角形
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    
                    # 检查凸包面积与轮廓面积的比值，三角形应该接近1
                    if hull_area > 0:
                        solidity = area / hull_area
                        # 三角形或多边形的solidity应该较高（>0.7）
                        if solidity > 0.7:
                            enemy_positions.append((cx, cy))
                            logger.debug(f"发现红色多边形敌人位置: ({cx}, {cy}), 面积: {area:.1f}, 顶点数: {vertices}, 紧密度: {solidity:.2f}")
            
            logger.info(f"识别到 {len(enemy_positions)} 个红色多边形敌人")
            return enemy_positions
            
        except Exception as e:
            logger.error(f"识别红色多边形敌人失败: {str(e)}")
            raise
    
    def detect_white_player(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        识别图像中的白色三角（自己）
        
        @param img: 输入图像（BGR格式）
        @return: 玩家位置(x, y)坐标，如果未找到返回None
        @raise Exception: 当图像处理失败时抛出异常
        """
        try:
            logger.info("开始识别白色三角（自己）")
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 创建白色掩码
            white_mask = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)
            
            # 形态学操作，去除噪声
            kernel = np.ones((5, 5), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("未找到白色三角（自己）")
                return None
            
            # 找到最大的轮廓（假设白色三角是最大的白色区域）
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(largest_contour)
                logger.info(f"找到自己位置: ({cx}, {cy}), 面积: {area}")
                return (cx, cy)
            else:
                logger.warning("未找到有效的白色三角（自己）")
                return None
                
        except Exception as e:
            logger.error(f"识别白色三角失败: {str(e)}")
            raise
    
    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        计算两点之间的欧氏距离
        
        @param pos1: 第一个点的坐标(x, y)
        @param pos2: 第二个点的坐标(x, y)
        @return: 两点之间的距离
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        distance = np.sqrt(dx * dx + dy * dy)
        return distance
    
    def find_nearest_enemy(self, player_pos: Tuple[int, int], 
                          enemy_positions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        找到离玩家最近的敌人
        
        @param player_pos: 玩家位置(x, y)
        @param enemy_positions: 敌人位置列表
        @return: 最近的敌人位置(x, y)，如果没有敌人返回None
        """
        if not enemy_positions:
            logger.warning("没有找到任何敌人")
            return None
        
        logger.info(f"计算 {len(enemy_positions)} 个敌人与自己的距离")
        min_distance = float('inf')
        nearest_enemy = None
        
        for enemy_pos in enemy_positions:
            distance = self.calculate_distance(player_pos, enemy_pos)
            logger.debug(f"敌人位置 {enemy_pos} 距离: {distance:.2f}")
            if distance < min_distance:
                min_distance = distance
                nearest_enemy = enemy_pos
        
        if nearest_enemy:
            logger.info(f"最近的敌人位置: {nearest_enemy}, 距离: {min_distance:.2f}")
        
        return nearest_enemy
    
    def random_delay(self, min_seconds: float = None, max_seconds: float = None) -> None:
        """
        随机延迟，模拟人类操作的不规律性
        
        @param min_seconds: 最小延迟时间（秒），默认使用类常量
        @param max_seconds: 最大延迟时间（秒），默认使用类常量
        """
        if min_seconds is None:
            min_seconds = self.MIN_RANDOM_DELAY
        if max_seconds is None:
            max_seconds = self.MAX_RANDOM_DELAY
        
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
        logger.debug(f"随机延迟: {delay:.3f}秒")
    
    def human_like_mouse_move(self, start_pos: Tuple[int, int], 
                              end_pos: Tuple[int, int]) -> None:
        """
        模拟人类鼠标移动，使用缓慢的曲线移动
        
        @param start_pos: 起始位置(x, y)
        @param end_pos: 目标位置(x, y)
        @raise Exception: 当鼠标移动失败时抛出异常
        """
        try:
            logger.info(f"缓慢移动鼠标从 {start_pos} 到 {end_pos}")
            
            # 计算距离
            distance = self.calculate_distance(start_pos, end_pos)
            
            # 根据距离计算移动时间（距离越远，时间越长）
            # 人类鼠标移动速度大约在 300-800 像素/秒
            base_speed = random.uniform(300, 800)  # 像素/秒
            move_duration = distance / base_speed
            
            # 限制在最小和最大时间范围内
            move_duration = max(self.MIN_MOVE_DURATION, 
                              min(move_duration, self.MAX_MOVE_DURATION))
            
            # 添加一些随机性
            move_duration *= random.uniform(0.9, 1.1)
            
            logger.info(f"鼠标移动距离: {distance:.2f}像素, 移动时间: {move_duration:.3f}秒")
            
            # 使用分段移动来模拟人类移动，添加缓动和随机性
            # 计算步数，每步约20-30ms
            step_interval = random.uniform(0.02, 0.03)  # 每步间隔20-30ms
            steps = max(15, int(move_duration / step_interval))
            actual_step_interval = move_duration / steps
            
            for i in range(steps + 1):
                # 使用easeInOut缓动函数，让移动更自然
                t = i / steps
                # easeInOut cubic: t < 0.5 ? 4t^3 : 1 - pow(-2t + 2, 3)/2
                if t < 0.5:
                    eased_t = 4 * t * t * t
                else:
                    eased_t = 1 - pow(-2 * t + 2, 3) / 2
                
                # 添加微小的随机偏移，模拟手部自然抖动（只在中间阶段添加）
                if 0.2 < t < 0.8:
                    noise_x = random.uniform(-1.5, 1.5)
                    noise_y = random.uniform(-1.5, 1.5)
                else:
                    noise_x = random.uniform(-0.5, 0.5)
                    noise_y = random.uniform(-0.5, 0.5)
                
                current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * eased_t + noise_x
                current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * eased_t + noise_y
                
                # 确保坐标在屏幕范围内
                current_x = max(0, min(current_x, self.SCREEN_WIDTH - 1))
                current_y = max(0, min(current_y, self.SCREEN_HEIGHT - 1))
                
                # 移动到当前点，使用很小的duration来保证平滑
                pyautogui.moveTo(int(current_x), int(current_y), duration=actual_step_interval)
            
            logger.info("鼠标移动完成")
            
        except Exception as e:
            logger.error(f"鼠标移动失败: {str(e)}")
            raise
    
    def open_map(self) -> bool:
        """
        按M键打开地图
        
        @return: 操作是否成功
        @raise Exception: 当键盘操作失败时抛出异常
        """
        try:
            logger.info("按M键打开地图")
            # 添加随机延迟，模拟人类按键前的思考时间
            self.random_delay(0.1, 0.3)
            pyautogui.press('m')
            # 等待地图完全打开，使用较长的延迟确保地图加载完成
            wait_time = random.uniform(1.0, 1.5)
            logger.info(f"等待地图完全打开（等待时间: {wait_time:.3f}秒）")
            time.sleep(wait_time)
            logger.info("地图已打开")
            return True
        except Exception as e:
            logger.error(f"打开地图失败: {str(e)}")
            raise
    
    def click_target_position(self, target_pos: Tuple[int, int], 
                              offset: Tuple[int, int] = (0, -50)) -> bool:
        """
        点击目标位置（敌人附近区域），使用缓慢移动模拟人类操作
        
        @param target_pos: 目标位置(x, y)
        @param offset: 偏移量，默认在敌人上方50像素处点击
        @return: 操作是否成功
        @raise Exception: 当鼠标操作失败时抛出异常
        """
        try:
            # 计算实际点击位置（敌人附近）
            click_x = target_pos[0] + offset[0]
            click_y = target_pos[1] + offset[1]
            
            # 添加小的随机偏移，让点击位置更自然
            random_offset_x = random.randint(-10, 10)
            random_offset_y = random.randint(-10, 10)
            click_x += random_offset_x
            click_y += random_offset_y
            
            # 确保坐标在屏幕范围内
            click_x = max(0, min(click_x, self.SCREEN_WIDTH - 1))
            click_y = max(0, min(click_y, self.SCREEN_HEIGHT - 1))
            
            # 获取当前鼠标位置
            current_pos = pyautogui.position()
            start_pos = (current_pos.x, current_pos.y)
            end_pos = (int(click_x), int(click_y))
            
            logger.info(f"准备点击目标位置: {end_pos}")
            
            # 缓慢移动鼠标到目标位置
            self.human_like_mouse_move(start_pos, end_pos)
            
            # 移动后稍作停顿，模拟人类瞄准时间
            self.random_delay(0.1, 0.2)
            
            # 执行点击
            logger.info("执行鼠标左键点击")
            pyautogui.click(button='left')
            
            # 点击后稍作停顿
            self.random_delay(0.2, 0.4)
            
            logger.info("点击完成，战舰开始自动导航")
            return True
        except Exception as e:
            logger.error(f"点击目标位置失败: {str(e)}")
            raise
    
    def draw_positions(self, img: np.ndarray, player_pos: Tuple[int, int],
                      enemy_positions: List[Tuple[int, int]], 
                      nearest_enemy: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        在图像上绘制玩家和敌人的位置关系
        
        @param img: 输入图像（BGR格式）
        @param player_pos: 玩家位置(x, y)
        @param enemy_positions: 所有敌人位置列表
        @param nearest_enemy: 最近的敌人位置(x, y)，如果提供会特别标记
        @return: 绘制后的图像
        @raise Exception: 当绘制失败时抛出异常
        """
        try:
            logger.info("开始绘制位置关系图")
            # 复制图像，避免修改原图
            result_img = img.copy()
            
            # 绘制所有敌人位置（红色圆圈）
            for enemy_pos in enemy_positions:
                x, y = enemy_pos
                # 绘制红色圆圈标记敌人
                cv2.circle(result_img, (x, y), 15, (0, 0, 255), 3)  # 红色圆圈
                cv2.circle(result_img, (x, y), 5, (0, 0, 255), -1)  # 红色实心圆
                # 添加文字标签
                cv2.putText(result_img, "Enemy", (x + 20, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 特别标记最近的敌人（黄色高亮）
            if nearest_enemy:
                x, y = nearest_enemy
                # 绘制黄色大圆圈标记最近敌人
                cv2.circle(result_img, (x, y), 25, (0, 255, 255), 4)  # 黄色外圈
                cv2.circle(result_img, (x, y), 15, (0, 0, 255), 3)  # 红色中圈
                cv2.circle(result_img, (x, y), 5, (0, 0, 255), -1)  # 红色实心圆
                # 添加"最近敌人"标签
                cv2.putText(result_img, "Nearest Enemy", (x + 30, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 绘制从玩家到最近敌人的连线
                if player_pos:
                    cv2.line(result_img, player_pos, nearest_enemy, (0, 255, 255), 2)
                    # 计算距离
                    distance = self.calculate_distance(player_pos, nearest_enemy)
                    # 在连线中点显示距离
                    mid_x = (player_pos[0] + nearest_enemy[0]) // 2
                    mid_y = (player_pos[1] + nearest_enemy[1]) // 2
                    cv2.putText(result_img, f"{distance:.1f}px", (mid_x, mid_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 绘制玩家位置（白色/青色三角，增强标记）
            if player_pos:
                x, y = player_pos
                
                # 绘制大圆圈背景（青色/绿色，更醒目）
                cv2.circle(result_img, (x, y), 30, (0, 255, 0), 4)  # 绿色外圈
                cv2.circle(result_img, (x, y), 25, (255, 255, 0), 3)  # 青色中圈
                
                # 绘制白色三角形标记玩家（更大更明显）
                triangle_size = 25
                pts = np.array([
                    [x, y - triangle_size],  # 顶点
                    [x - triangle_size, y + triangle_size],  # 左下
                    [x + triangle_size, y + triangle_size]   # 右下
                ], np.int32)
                cv2.fillPoly(result_img, [pts], (255, 255, 255))  # 白色填充
                cv2.polylines(result_img, [pts], True, (0, 255, 255), 4)  # 黄色边框，更粗
                
                # 在中心点绘制小圆点
                cv2.circle(result_img, (x, y), 5, (0, 0, 255), -1)  # 红色中心点
                
                # 添加文字标签（更大更醒目）
                label_text = "Player (You)"
                label_x = x + 35
                label_y = y - 30
                
                # 绘制文字背景（黑色半透明）
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(result_img, 
                             (label_x - 5, label_y - text_height - 5),
                             (label_x + text_width + 5, label_y + baseline + 5),
                             (0, 0, 0), -1)
                
                # 绘制文字
                cv2.putText(result_img, label_text, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 添加坐标信息
                coord_text = f"({x}, {y})"
                coord_y = y + 40
                (coord_width, coord_height), baseline = cv2.getTextSize(
                    coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(result_img,
                             (x - coord_width // 2 - 5, coord_y - coord_height - 5),
                             (x + coord_width // 2 + 5, coord_y + baseline + 5),
                             (0, 0, 0), -1)
                cv2.putText(result_img, coord_text, 
                           (x - coord_width // 2, coord_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                logger.info(f"已标记玩家位置: ({x}, {y})")
            
            # 绘制所有敌人到玩家的距离线（浅色虚线）
            if player_pos:
                for enemy_pos in enemy_positions:
                    if enemy_pos != nearest_enemy:  # 最近敌人已经画过了
                        # 绘制浅灰色虚线
                        cv2.line(result_img, player_pos, enemy_pos, (128, 128, 128), 1, 
                                cv2.LINE_AA)
                        # 计算并显示距离
                        distance = self.calculate_distance(player_pos, enemy_pos)
                        mid_x = (player_pos[0] + enemy_pos[0]) // 2
                        mid_y = (player_pos[1] + enemy_pos[1]) // 2
                        cv2.putText(result_img, f"{distance:.0f}", (mid_x, mid_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # 绘制信息面板
            self._draw_info_panel(result_img, player_pos, enemy_positions, nearest_enemy)
            
            logger.info("位置关系图绘制完成")
            return result_img
            
        except Exception as e:
            logger.error(f"绘制位置关系图失败: {str(e)}")
            raise
    
    def _draw_info_panel(self, img: np.ndarray, player_pos: Tuple[int, int],
                        enemy_positions: List[Tuple[int, int]], 
                        nearest_enemy: Optional[Tuple[int, int]]) -> None:
        """
        在图像上绘制信息面板，显示统计信息
        
        @param img: 输入图像（BGR格式）
        @param player_pos: 玩家位置
        @param enemy_positions: 所有敌人位置列表
        @param nearest_enemy: 最近的敌人位置
        """
        try:
            # 计算面板位置（左上角）
            panel_x = 10
            panel_y = 10
            panel_width = 300
            panel_height = 150
            
            # 绘制半透明背景
            overlay = img.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            # 绘制边框
            cv2.rectangle(img, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (255, 255, 255), 2)
            
            # 显示信息
            y_offset = 30
            line_height = 25
            
            # 标题
            cv2.putText(img, "Position Info", (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height
            
            # 玩家位置
            cv2.putText(img, f"Player: ({player_pos[0]}, {player_pos[1]})", 
                       (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # 敌人数量
            cv2.putText(img, f"Enemies Found: {len(enemy_positions)}", 
                       (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += line_height
            
            # 最近敌人距离
            if nearest_enemy:
                distance = self.calculate_distance(player_pos, nearest_enemy)
                cv2.putText(img, f"Nearest Distance: {distance:.1f}px", 
                           (panel_x + 10, panel_y + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += line_height
                cv2.putText(img, f"Target: ({nearest_enemy[0]}, {nearest_enemy[1]})", 
                           (panel_x + 10, panel_y + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        except Exception as e:
            logger.warning(f"绘制信息面板失败: {str(e)}")
    
    def save_visualization(self, img: np.ndarray, filename: Optional[str] = None) -> str:
        """
        保存可视化结果图像
        
        @param img: 要保存的图像（BGR格式）
        @param filename: 文件名，如果为None则自动生成
        @return: 保存的文件路径
        @raise Exception: 当保存失败时抛出异常
        """
        try:
            if filename is None:
                # 自动生成文件名，包含时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"wows_position_{timestamp}.png"
            
            # 确保输出目录存在
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"创建输出目录: {output_dir}")
            
            filepath = os.path.join(output_dir, filename)
            # OpenCV保存图像（BGR格式）
            cv2.imwrite(filepath, img)
            logger.info(f"可视化结果已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存可视化结果失败: {str(e)}")
            raise
    
    def run(self) -> bool:
        """
        执行完整的自动导航流程
        
        @return: 操作是否成功
        @raise Exception: 当任何步骤失败时抛出异常
        """
        try:
            logger.info("=" * 50)
            logger.info("开始执行自动导航流程")
            logger.info("=" * 50)
            
            # 步骤1: 捕获屏幕
            logger.info("步骤1: 捕获屏幕截图")
            img = self.capture_screen()
            self.screenshot_img = img.copy()  # 保存原始截图用于可视化
            
            # 步骤2: 识别玩家位置
            logger.info("步骤2: 识别玩家位置（白色三角）")
            self.player_pos = self.detect_white_player(img)
            if self.player_pos is None:
                logger.error("未找到玩家位置，无法继续")
                return False
            
            # 步骤3: 识别敌人位置
            logger.info("步骤3: 识别敌人位置（红色）")
            self.enemy_positions = self.detect_red_enemies(img)
            if not self.enemy_positions:
                logger.warning("未找到任何敌人")
                return False
            
            # 步骤4: 找到最近的敌人
            logger.info("步骤4: 计算距离，找到最近的敌人")
            nearest_enemy = self.find_nearest_enemy(self.player_pos, self.enemy_positions)
            self.nearest_enemy = nearest_enemy
            if nearest_enemy is None:
                logger.warning("未找到最近的敌人")
                return False
            
            # 步骤4.5: 绘制位置关系图
            logger.info("步骤4.5: 绘制位置关系图")
            try:
                visualized_img = self.draw_positions(
                    self.screenshot_img, 
                    self.player_pos, 
                    self.enemy_positions, 
                    nearest_enemy
                )
                # 保存可视化结果
                self.save_visualization(visualized_img)
                logger.info("位置关系图已生成并保存")
            except Exception as e:
                logger.warning(f"绘制位置关系图失败，继续执行: {str(e)}")
            
            # 步骤5: 打开地图
            logger.info("步骤5: 打开地图")
            if not self.open_map():
                logger.error("打开地图失败")
                return False
            
            # 步骤5.5: 延迟截图，确保地图完全加载后再截图
            logger.info("步骤5.5: 延迟截图，等待地图完全加载")
            screenshot_delay = random.uniform(0.5, 0.8)
            logger.info(f"延迟 {screenshot_delay:.3f} 秒后重新截图")
            time.sleep(screenshot_delay)
            
            # 重新捕获屏幕（地图打开后的截图）
            logger.info("重新捕获屏幕截图（地图已打开）")
            img = self.capture_screen()
            self.screenshot_img = img.copy()  # 更新截图用于可视化
            
            # 重新识别位置（在地图上）
            logger.info("在地图上重新识别位置")
            self.player_pos = self.detect_white_player(img)
            if self.player_pos is None:
                logger.warning("在地图上未找到玩家位置，使用之前的位置")
            else:
                logger.info(f"在地图上找到玩家位置: {self.player_pos}")
            
            self.enemy_positions = self.detect_red_enemies(img)
            if not self.enemy_positions:
                logger.warning("在地图上未找到任何敌人，使用之前识别的敌人")
            else:
                logger.info(f"在地图上找到 {len(self.enemy_positions)} 个敌人")
                # 重新计算最近的敌人
                if self.player_pos:
                    nearest_enemy = self.find_nearest_enemy(self.player_pos, self.enemy_positions)
                    if nearest_enemy:
                        self.nearest_enemy = nearest_enemy
                        logger.info(f"在地图上重新计算，最近敌人: {nearest_enemy}")
            
            # 更新可视化图像
            if self.player_pos and self.enemy_positions:
                try:
                    visualized_img = self.draw_positions(
                        self.screenshot_img, 
                        self.player_pos, 
                        self.enemy_positions, 
                        self.nearest_enemy
                    )
                    self.save_visualization(visualized_img)
                    logger.info("位置关系图已更新并保存")
                except Exception as e:
                    logger.warning(f"更新位置关系图失败: {str(e)}")
            
            # 步骤6: 点击最近的敌人附近区域
            logger.info("步骤6: 点击最近的敌人附近区域")
            # 在敌人前方一点的位置点击（偏移量可以根据需要调整）
            if not self.click_target_position(nearest_enemy, offset=(0, -80)):
                logger.error("点击目标位置失败")
                return False
            
            logger.info("=" * 50)
            logger.info("自动导航流程执行完成")
            logger.info("=" * 50)
            return True
            
        except Exception as e:
            logger.error(f"执行自动导航流程失败: {str(e)}")
            raise


def main():
    """主函数"""
    try:
        logger.info("程序启动")
        navigator = WowsAutoNavigate()
        success = navigator.run()
        
        if success:
            logger.info("程序执行成功")
            return 0
        else:
            logger.warning("程序执行完成，但可能未找到目标")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

