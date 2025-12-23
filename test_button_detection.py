"""
按钮检测测试脚本
用于测试橙色"加入战斗"按钮的检测是否正常工作

@author seven
@since 2025-12-23
"""

import cv2
import numpy as np
import pyautogui
from pathlib import Path

# 颜色范围
ORANGE_LOWER = np.array([10, 100, 100])
ORANGE_UPPER = np.array([25, 255, 255])

# 按钮区域
JOIN_BATTLE_BUTTON_REGION = (650, 0, 850, 50)


def test_button_detection():
    """
    测试橙色按钮检测
    
    @since 2025-12-23
    """
    print("=" * 60)
    print("橙色按钮检测测试")
    print("=" * 60)
    print()
    print("请确保：")
    print("1. 游戏已打开并处于港口界面")
    print("2. 游戏窗口在最前面，没有被遮挡")
    print("3. 游戏分辨率为 1920x1080")
    print()
    input("按回车键开始测试...")
    
    print()
    print("正在截取屏幕...")
    
    # 截取屏幕
    screenshot = pyautogui.screenshot()
    img = np.array(screenshot)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    print(f"屏幕尺寸: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
    
    # 提取按钮区域
    x1, y1, x2, y2 = JOIN_BATTLE_BUTTON_REGION
    button_region = img_bgr[y1:y2, x1:x2]
    
    print(f"按钮区域: ({x1}, {y1}) 到 ({x2}, {y2})")
    print(f"按钮区域尺寸: {button_region.shape[1]}x{button_region.shape[0]}")
    
    # 转换为HSV并检测橙色
    hsv = cv2.cvtColor(button_region, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    orange_pixels = cv2.countNonZero(orange_mask)
    
    print()
    print("检测结果:")
    print(f"  橙色像素数: {orange_pixels}")
    print(f"  区域总像素: {button_region.shape[0] * button_region.shape[1]}")
    print(f"  橙色占比: {orange_pixels / (button_region.shape[0] * button_region.shape[1]) * 100:.2f}%")
    print()
    
    # 判断结果
    if orange_pixels > 100:
        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2
        print("✅ 检测成功！")
        print(f"  按钮中心位置: ({center_x}, {center_y})")
        
        # 在完整图像上标记
        result_img = img_bgr.copy()
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(result_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 保存结果
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / "test_full_screen.png"), result_img)
        cv2.imwrite(str(output_dir / "test_button_region.png"), button_region)
        cv2.imwrite(str(output_dir / "test_orange_mask.png"), orange_mask)
        
        print()
        print("结果图片已保存到 debug_output/ 目录:")
        print("  - test_full_screen.png (完整屏幕，标记了按钮位置)")
        print("  - test_button_region.png (按钮区域)")
        print("  - test_orange_mask.png (橙色检测掩码)")
        
    else:
        print("❌ 检测失败！")
        print()
        print("可能的原因:")
        print("1. 当前不在港口界面")
        print("2. 按钮区域设置不正确")
        print("3. 橙色颜色范围不匹配")
        print()
        print("建议:")
        print("1. 确认游戏界面状态")
        print("2. 查看保存的图片，手动确认按钮位置")
        print("3. 根据实际情况调整参数")
        
        # 保存截图用于调试
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / "test_full_screen.png"), img_bgr)
        cv2.imwrite(str(output_dir / "test_button_region.png"), button_region)
        cv2.imwrite(str(output_dir / "test_orange_mask.png"), orange_mask)
        
        print()
        print("调试图片已保存到 debug_output/ 目录")
    
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)


def test_with_sample_image():
    """
    使用样例图片测试
    
    @since 2025-12-23
    """
    print("=" * 60)
    print("使用样例图片测试")
    print("=" * 60)
    
    sample_path = Path("img/6dbb35aad67bf33c816d33c38e1978e1.png")
    
    if not sample_path.exists():
        print(f"样例图片不存在: {sample_path}")
        return
    
    print(f"读取图片: {sample_path}")
    img = cv2.imread(str(sample_path))
    
    if img is None:
        print("图片读取失败")
        return
    
    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")
    
    # 提取按钮区域
    x1, y1, x2, y2 = JOIN_BATTLE_BUTTON_REGION
    button_region = img[y1:y2, x1:x2]
    
    # 转换为HSV并检测橙色
    hsv = cv2.cvtColor(button_region, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    orange_pixels = cv2.countNonZero(orange_mask)
    
    print()
    print("检测结果:")
    print(f"  橙色像素数: {orange_pixels}")
    print()
    
    if orange_pixels > 100:
        print("✅ 在样例图片中检测到按钮！")
        
        # 标记按钮
        result_img = img.copy()
        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(result_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 保存结果
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "test_sample_result.png"), result_img)
        
        print(f"结果已保存到: debug_output/test_sample_result.png")
    else:
        print("❌ 未检测到按钮")
    
    print()
    print("=" * 60)


def main():
    """主函数"""
    print()
    print("橙色按钮检测测试工具")
    print()
    print("请选择测试方式:")
    print("1. 实时截屏测试（需要打开游戏）")
    print("2. 使用样例图片测试")
    print()
    
    choice = input("请输入选项 (1 或 2): ").strip()
    
    print()
    
    if choice == "1":
        test_button_detection()
    elif choice == "2":
        test_with_sample_image()
    else:
        print("无效选项")


if __name__ == "__main__":
    main()

