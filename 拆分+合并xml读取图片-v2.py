import os
import sys
import json
import logging
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, NamedTuple
from pathlib import Path
import time
from dataclasses import dataclass
from functools import lru_cache
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('image_processor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """处理统计信息"""
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = 0
    
    def add_success(self):
        self.successful += 1
    
    def add_failure(self):
        self.failed += 1
    
    def add_skip(self):
        self.skipped += 1
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        return (self.successful / self.total * 100) if self.total > 0 else 0

class ImageInfo(NamedTuple):
    """图片信息"""
    name: str
    x: int
    y: int
    width: int
    height: int

class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        'max_workers': min(os.cpu_count() or 4, 8),
        'supported_formats': ['.png', '.webp', '.jpg', '.jpeg', '.bmp', '.tiff'],
        'output_format': 'PNG',
        'quality': 100,
        'create_backup': True,
        'validate_dimensions': True,
        'auto_fix_xml': True
    }
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """加载配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置
                    return {**self.DEFAULT_CONFIG, **config}
            except Exception as e:
                logger.warning(f"配置文件加载失败，使用默认配置: {e}")
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value

class ImageProcessor:
    """优化后的图片处理类"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self._xml_cache = {}  # XML解析缓存
        
    @staticmethod
    @lru_cache(maxsize=128)
    def normalize_filename(filename: str) -> str:
        """标准化文件名（带缓存）"""
        return re.sub(r'[^\w.]', '', filename.lower())

    def find_matching_files(self, folder_path: Path) -> List[Tuple[Path, Path]]:
        """智能查找匹配的图片和XML文件对"""
        img_exts = tuple(self.config.get('supported_formats'))
        
        # 获取所有文件
        all_files = list(folder_path.iterdir())
        image_files = [f for f in all_files if f.suffix.lower() in img_exts and f.is_file()]
        xml_files = [f for f in all_files if f.suffix.lower() == '.xml' and f.is_file()]
        
        if not image_files:
            raise ValueError(f"未找到支持的图片文件 (支持格式: {', '.join(img_exts)})")
        if not xml_files:
            raise ValueError("未找到XML文件")
        
        # 创建匹配映射
        xml_map = {}
        for xml_file in xml_files:
            base_name = self.normalize_filename(xml_file.stem)
            xml_map[base_name] = xml_file
        
        # 查找匹配对
        pairs = []
        unmatched = []
        
        for img_file in image_files:
            base_name = self.normalize_filename(img_file.stem)
            if base_name in xml_map:
                pairs.append((img_file, xml_map[base_name]))
            else:
                unmatched.append(img_file.name)
        
        if unmatched:
            logger.warning(f"以下图片文件未找到匹配的XML: {', '.join(unmatched)}")
        
        return pairs

    def fix_xml_structure(self, xml_content: str) -> str:
        """修复XML结构（优化版）"""
        if not xml_content.strip():
            raise ValueError("XML内容为空")
        
        # 移除BOM和多余空白
        xml_content = xml_content.strip().lstrip('\ufeff')
        
        # 添加XML声明
        if not xml_content.startswith('<?xml'):
            xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_content
        
        # 检查是否需要根元素
        try:
            ET.fromstring(xml_content)
            return xml_content
        except ET.ParseError:
            # 尝试添加根元素
            content_without_declaration = re.sub(r'<\?xml.*?\?>\s*', '', xml_content)
            return f'<?xml version="1.0" encoding="UTF-8"?>\n<root>{content_without_declaration}</root>'

    def parse_xml(self, xml_path: Path) -> Tuple[ET.Element, List[ImageInfo]]:
        """解析XML并提取图片信息（带缓存和验证）"""
        cache_key = str(xml_path)
        
        # 检查缓存
        if cache_key in self._xml_cache:
            xml_stat = xml_path.stat()
            cached_data, cached_mtime = self._xml_cache[cache_key]
            if cached_mtime == xml_stat.st_mtime:
                return cached_data
        
        try:
            # 尝试多种编码
            encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
            content = None
            
            for encoding in encodings:
                try:
                    with open(xml_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError("无法解码XML文件，尝试了多种编码方式")
            
            # 修复XML结构
            if self.config.get('auto_fix_xml'):
                content = self.fix_xml_structure(content)
            
            # 解析XML
            root = ET.fromstring(content)
            
            # 提取图片信息
            image_infos = []
            image_nodes = root.findall('.//Image') or root.findall('Image')
            
            if not image_nodes:
                # 尝试其他可能的节点名
                alternative_names = ['image', 'Img', 'Picture', 'pic']
                for name in alternative_names:
                    image_nodes = root.findall(f'.//{name}') or root.findall(name)
                    if image_nodes:
                        break
            
            for i, node in enumerate(image_nodes):
                try:
                    attrs = node.attrib
                    name = attrs.get('name', f'unnamed_{i}.png')
                    
                    # 确保PNG扩展名
                    if not name.lower().endswith('.png'):
                        name = Path(name).stem + '.png'
                    
                    # 解析坐标和尺寸
                    x = int(float(attrs.get('x', 0)))
                    y = int(float(attrs.get('y', 0)))
                    w = int(float(attrs.get('w', attrs.get('width', 0))))
                    h = int(float(attrs.get('h', attrs.get('height', 0))))
                    
                    if w <= 0 or h <= 0:
                        logger.warning(f"跳过无效尺寸的图片: {name} ({w}x{h})")
                        continue
                    
                    image_infos.append(ImageInfo(name, x, y, w, h))
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"解析图片节点 #{i+1} 失败: {e}")
                    continue
            
            if not image_infos:
                raise ValueError("XML中未找到有效的图片信息")
            
            # 缓存结果
            result = (root, image_infos)
            xml_stat = xml_path.stat()
            self._xml_cache[cache_key] = (result, xml_stat.st_mtime)
            
            return result
            
        except Exception as e:
            raise ValueError(f"XML解析失败 ({xml_path.name}): {str(e)}")

    def validate_image_bounds(self, img_size: Tuple[int, int], image_info: ImageInfo) -> bool:
        """验证图片裁剪边界"""
        img_w, img_h = img_size
        x, y, w, h = image_info.x, image_info.y, image_info.width, image_info.height
        
        if x < 0 or y < 0:
            return False
        if x + w > img_w or y + h > img_h:
            return False
        if w <= 0 or h <= 0:
            return False
            
        return True

    def split_image(self, image_path: Path, xml_path: Path, output_folder: Path) -> ProcessingStats:
        """拆分图片（优化版）"""
        stats = ProcessingStats()
        stats.start_time = time.time()
        
        try:
            logger.info(f"开始拆分: {image_path.name}")
            
            # 打开并优化图片
            with Image.open(image_path) as img:
                # 自动修正图片方向
                img = ImageOps.exif_transpose(img)
                img = img.convert('RGBA')
                img_size = img.size
            
            # 解析XML
            root, image_infos = self.parse_xml(xml_path)
            stats.total = len(image_infos)
            
            # 创建输出目录
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # 保存元数据
            metadata = {
                'original_image': str(image_path),
                'original_size': img_size,
                'xml_file': str(xml_path),
                'timestamp': time.time(),
                'images': []
            }
            
            # 处理每个图片区域
            for info in image_infos:
                try:
                    # 验证边界
                    if self.config.get('validate_dimensions') and not self.validate_image_bounds(img_size, info):
                        logger.warning(f"图片 {info.name} 边界超出原图范围，跳过")
                        stats.add_skip()
                        continue
                    
                    # 裁剪图片
                    box = (info.x, info.y, info.x + info.width, info.y + info.height)
                    cropped = img.crop(box)
                    
                    # 保存图片
                    output_path = output_folder / info.name
                    save_kwargs = {
                        'format': self.config.get('output_format', 'PNG'),
                        'optimize': True
                    }
                    
                    if self.config.get('output_format') == 'PNG':
                        save_kwargs['compress_level'] = 6  # 平衡压缩率和速度
                    
                    cropped.save(output_path, **save_kwargs)
                    
                    # 记录元数据
                    metadata['images'].append({
                        'name': info.name,
                        'original_pos': [info.x, info.y],
                        'size': [info.width, info.height]
                    })
                    
                    stats.add_success()
                    logger.debug(f"✓ 已保存: {info.name} ({info.width}x{info.height})")
                    
                except Exception as e:
                    logger.error(f"处理 {info.name} 时出错: {e}")
                    stats.add_failure()
            
            # 保存元数据文件
            metadata_path = output_folder / '_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"拆分完成: {stats.successful}/{stats.total} 成功, "
                       f"用时 {stats.elapsed_time:.2f}s")
            
            return stats
            
        except Exception as e:
            logger.error(f"拆分失败 ({image_path.name}): {e}")
            stats.add_failure()
            return stats

    def merge_images(self, image_path: Path, xml_path: Path, modified_folder: Path) -> Optional[Path]:
        """合并图片（优化版）"""
        try:
            logger.info(f"开始合并: {image_path.name}")
            start_time = time.time()
            
            # 读取元数据
            metadata_path = modified_folder / '_metadata.json'
            if not metadata_path.exists():
                raise FileNotFoundError("找不到元数据文件，请确保使用本工具拆分的图片")
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 打开原始图片
            with Image.open(image_path) as original_img:
                original_img = ImageOps.exif_transpose(original_img).convert('RGBA')
                new_img = original_img.copy()
            
            # 处理统计
            total_images = len(metadata['images'])
            merged_count = 0
            missing_count = 0
            modified_count = 0
            
            for img_info in metadata['images']:
                name = img_info['name']
                x, y = img_info['original_pos']
                original_w, original_h = img_info['size']
                
                modified_path = modified_folder / name
                if not modified_path.exists():
                    logger.warning(f"未找到修改后的图片: {name}")
                    missing_count += 1
                    continue
                
                try:
                    # 打开修改后的图片
                    with Image.open(modified_path) as modified_img:
                        modified_img = modified_img.convert('RGBA')
                        w, h = modified_img.size
                        
                        # 检查是否被修改过
                        if w != original_w or h != original_h:
                            logger.info(f"图片 {name} 尺寸已变化: {original_w}x{original_h} -> {w}x{h}")
                            modified_count += 1
                            
                            # 如果尺寸过大，自动调整
                            if w > original_w or h > original_h:
                                logger.warning(f"调整 {name} 尺寸至原始大小")
                                modified_img = modified_img.resize((original_w, original_h), Image.LANCZOS)
                        
                        # 粘贴到新图片
                        new_img.paste(modified_img, (x, y), modified_img)
                        merged_count += 1
                        
                except Exception as e:
                    logger.error(f"处理 {name} 时出错: {e}")
                    missing_count += 1
            
            if merged_count == 0:
                logger.warning("没有成功合并任何图片")
                return None
            
            # 生成输出路径
            output_path = image_path.parent / f"{image_path.stem}_merged.png"
            counter = 1
            while output_path.exists():
                output_path = image_path.parent / f"{image_path.stem}_merged_{counter}.png"
                counter += 1
            
            # 保存合并后的图片
            save_kwargs = {
                'format': 'PNG',
                'optimize': True,
                'compress_level': 6
            }
            new_img.save(output_path, **save_kwargs)
            
            elapsed_time = time.time() - start_time
            logger.info(f"合并完成: {output_path.name}")
            logger.info(f"统计: 成功 {merged_count}/{total_images}, "
                       f"缺失 {missing_count}, 修改 {modified_count}, "
                       f"用时 {elapsed_time:.2f}s")
            
            return output_path
            
        except Exception as e:
            logger.error(f"合并失败: {e}")
            return None

class BatchProcessor:
    """优化后的批量处理类"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.processor = ImageProcessor(config_manager)
    
    def batch_split_images(self, folder_path: Path) -> ProcessingStats:
        """批量拆分图片（优化版）"""
        total_stats = ProcessingStats()
        total_stats.start_time = time.time()
        
        try:
            logger.info(f"开始批量拆分: {folder_path}")
            
            # 查找匹配的文件对
            pairs = self.processor.find_matching_files(folder_path)
            total_stats.total = len(pairs)
            
            if not pairs:
                logger.warning("未找到任何匹配的图片和XML文件对")
                return total_stats
            
            logger.info(f"找到 {len(pairs)} 对文件，开始处理...")
            
            # 多线程处理
            max_workers = self.config.get('max_workers')
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_pair = {}
                for img_path, xml_path in pairs:
                    output_dir = folder_path / f"{img_path.stem}_split"
                    future = executor.submit(self.processor.split_image, img_path, xml_path, output_dir)
                    future_to_pair[future] = (img_path, xml_path)
                
                # 收集结果
                for future in as_completed(future_to_pair):
                    img_path, xml_path = future_to_pair[future]
                    try:
                        stats = future.result()
                        total_stats.successful += stats.successful
                        total_stats.failed += stats.failed
                        total_stats.skipped += stats.skipped
                    except Exception as e:
                        logger.error(f"处理 {img_path.name} 时发生异常: {e}")
                        total_stats.add_failure()
            
            logger.info(f"批量拆分完成: 成功 {total_stats.successful}, "
                       f"失败 {total_stats.failed}, 跳过 {total_stats.skipped}, "
                       f"用时 {total_stats.elapsed_time:.2f}s")
            
            return total_stats
            
        except Exception as e:
            logger.error(f"批量拆分失败: {e}")
            return total_stats
    
    def batch_merge_images(self, folder_path: Path) -> ProcessingStats:
        """批量合并图片（优化版）"""
        stats = ProcessingStats()
        stats.start_time = time.time()
        
        try:
            logger.info(f"开始批量合并: {folder_path}")
            
            # 查找所有拆分文件夹
            split_folders = [f for f in folder_path.iterdir() 
                           if f.is_dir() and f.name.endswith('_split')]
            
            if not split_folders:
                logger.warning("未找到任何拆分文件夹（应以_split结尾）")
                return stats
            
            # 准备合并任务
            merge_tasks = []
            img_exts = tuple(self.config.get('supported_formats'))
            
            for split_folder in split_folders:
                base_name = split_folder.name.replace('_split', '')
                
                # 查找原始图片和XML
                img_file = None
                xml_file = None
                
                for file_path in folder_path.iterdir():
                    if not file_path.is_file():
                        continue
                        
                    if file_path.stem == base_name:
                        if file_path.suffix.lower() in img_exts:
                            img_file = file_path
                        elif file_path.suffix.lower() == '.xml':
                            xml_file = file_path
                
                if img_file and xml_file:
                    merge_tasks.append((img_file, xml_file, split_folder))
                else:
                    logger.warning(f"未找到 {base_name} 的原始文件")
            
            stats.total = len(merge_tasks)
            
            if not merge_tasks:
                logger.warning("未找到任何可合并的任务")
                return stats
            
            logger.info(f"找到 {len(merge_tasks)} 个合并任务，开始处理...")
            
            # 多线程处理
            max_workers = self.config.get('max_workers')
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {}
                for img_path, xml_path, split_dir in merge_tasks:
                    future = executor.submit(self.processor.merge_images, img_path, xml_path, split_dir)
                    future_to_task[future] = (img_path, split_dir)
                
                for future in as_completed(future_to_task):
                    img_path, split_dir = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            stats.add_success()
                        else:
                            stats.add_failure()
                    except Exception as e:
                        logger.error(f"合并 {img_path.name} 时发生异常: {e}")
                        stats.add_failure()
            
            logger.info(f"批量合并完成: 成功 {stats.successful}/{stats.total}, "
                       f"用时 {stats.elapsed_time:.2f}s")
            
            return stats
            
        except Exception as e:
            logger.error(f"批量合并失败: {e}")
            return stats

class InteractiveUI:
    """交互式用户界面"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.batch_processor = BatchProcessor(self.config)
        self.image_processor = ImageProcessor(self.config)
    
    def show_banner(self):
        """显示程序横幅"""
        print("🛠️" + "=" * 60)
        print("          图片拆分与合并工具 v3.0 (全面优化版)")
        print("=" * 62)
        print("✨ 特性: 智能匹配 | 多线程处理 | 自动修复 | 进度跟踪")
        print("🔧 优化: 内存效率 | 错误恢复 | 配置管理 | 日志记录")
        print("=" * 62)
    
    def show_menu(self):
        """显示主菜单"""
        print("\n📋 操作菜单:")
        print("1. 🔄 单个文件拆分")
        print("2. 📁 批量拆分文件夹")
        print("3. 🔗 单个文件合并")
        print("4. 🗂️  批量合并文件夹")
        print("5. ⚙️  配置设置")
        print("6. 📊 查看日志")
        print("0. 🚪 退出程序")
        print("=" * 40)
    
    def get_user_input(self, prompt: str, validation_func=None) -> str:
        """获取用户输入并验证"""
        while True:
            try:
                value = input(f"📝 {prompt}: ").strip().strip('"')
                if not value:
                    print("❌ 输入不能为空，请重新输入")
                    continue
                    
                if validation_func and not validation_func(value):
                    print("❌ 输入格式不正确，请重新输入")
                    continue
                    
                return value
            except KeyboardInterrupt:
                print("\n\n👋 用户取消操作")
                return ""
    
    def path_validator(self, path_str: str) -> bool:
        """路径验证器"""
        path = Path(path_str)
        return path.exists()
    
    def show_config_menu(self):
        """显示配置菜单"""
        while True:
            print("\n⚙️ 配置设置:")
            print(f"1. 线程数: {self.config.get('max_workers')}")
            print(f"2. 输出格式: {self.config.get('output_format')}")
            print(f"3. 创建备份: {'是' if self.config.get('create_backup') else '否'}")
            print(f"4. 验证尺寸: {'是' if self.config.get('validate_dimensions') else '否'}")
            print(f"5. 自动修复XML: {'是' if self.config.get('auto_fix_xml') else '否'}")
            print("6. 💾 保存配置")
            print("0. 🔙 返回主菜单")
            
            choice = input("\n选择配置项 (0-6): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                try:
                    workers = int(input(f"输入线程数 (当前: {self.config.get('max_workers')}): "))
                    if 1 <= workers <= 32:
                        self.config.set('max_workers', workers)
                        print("✅ 线程数已更新")
                    else:
                        print("❌ 线程数应在1-32之间")
                except ValueError:
                    print("❌ 请输入有效数字")
            elif choice == '2':
                formats = ['PNG', 'JPEG', 'WEBP']
                print("支持的格式:", ', '.join(formats))
                fmt = input("输入输出格式: ").upper()
                if fmt in formats:
                    self.config.set('output_format', fmt)
                    print("✅ 输出格式已更新")
                else:
                    print("❌ 不支持的格式")
            elif choice in ['3', '4', '5']:
                key_map = {'3': 'create_backup', '4': 'validate_dimensions', '5': 'auto_fix_xml'}
                key = key_map[choice]
                current = self.config.get(key)
                new_value = not current
                self.config.set(key, new_value)
                print(f"✅ 已{'启用' if new_value else '禁用'}")
            elif choice == '6':
                self.config.save_config()
                print("✅ 配置已保存")
    
    def show_progress(self, stats: ProcessingStats):
        """显示处理进度"""
        if stats.total > 0:
            progress = (stats.successful + stats.failed + stats.skipped) / stats.total * 100
            print(f"📊 进度: {progress:.1f}% | "
                  f"成功: {stats.successful} | "
                  f"失败: {stats.failed} | "
                  f"跳过: {stats.skipped}")
    
    def show_log(self):
        """显示日志文件内容"""
        log_file = Path('image_processor.log')
        if not log_file.exists():
            print("❌ 日志文件不存在")
            return
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"\n📊 日志文件 ({log_file.name}) - 最近50行:")
            print("=" * 60)
            
            # 显示最后50行
            for line in lines[-50:]:
                print(line.rstrip())
            
            print("=" * 60)
            print(f"💡 完整日志请查看: {log_file.absolute()}")
            
        except Exception as e:
            print(f"❌ 读取日志文件失败: {e}")
    
    def run(self):
        """运行主程序"""
        self.show_banner()
        
        while True:
            try:
                self.show_menu()
                choice = input("🎯 请选择操作 (0-6): ").strip()
                
                if choice == '0':
                    print("👋 感谢使用，再见！")
                    self.config.save_config()
                    break
                    
                elif choice == '1':
                    # 单个文件拆分
                    img_path = self.get_user_input("图片文件路径", self.path_validator)
                    if not img_path:
                        continue
                        
                    xml_path = self.get_user_input("XML文件路径", self.path_validator)
                    if not xml_path:
                        continue
                    
                    img_path = Path(img_path)
                    xml_path = Path(xml_path)
                    output_dir = img_path.parent / f"{img_path.stem}_split"
                    
                    print("🚀 开始拆分图片...")
                    stats = self.image_processor.split_image(img_path, xml_path, output_dir)
                    
                    print("\n📊 拆分结果:")
                    print(f"✅ 成功: {stats.successful}")
                    print(f"❌ 失败: {stats.failed}")
                    print(f"⏭️ 跳过: {stats.skipped}")
                    print(f"⏱️ 用时: {stats.elapsed_time:.2f}s")
                    print(f"📁 输出目录: {output_dir}")
                    
                elif choice == '2':
                    # 批量拆分文件夹
                    folder_path = self.get_user_input("文件夹路径", self.path_validator)
                    if not folder_path:
                        continue
                    
                    folder_path = Path(folder_path)
                    print("🚀 开始批量拆分...")
                    stats = self.batch_processor.batch_split_images(folder_path)
                    
                    print("\n📊 批量拆分结果:")
                    print(f"📁 处理文件夹: {folder_path}")
                    print(f"✅ 成功图片: {stats.successful}")
                    print(f"❌ 失败图片: {stats.failed}")
                    print(f"⏭️ 跳过图片: {stats.skipped}")
                    print(f"📈 成功率: {stats.success_rate:.1f}%")
                    print(f"⏱️ 总用时: {stats.elapsed_time:.2f}s")
                    
                elif choice == '3':
                    # 单个文件合并
                    img_path = self.get_user_input("原始图片文件路径", self.path_validator)
                    if not img_path:
                        continue
                        
                    xml_path = self.get_user_input("XML文件路径", self.path_validator)
                    if not xml_path:
                        continue
                        
                    split_folder = self.get_user_input("拆分图片文件夹路径", self.path_validator)
                    if not split_folder:
                        continue
                    
                    img_path = Path(img_path)
                    xml_path = Path(xml_path)
                    split_folder = Path(split_folder)
                    
                    print("🚀 开始合并图片...")
                    result = self.image_processor.merge_images(img_path, xml_path, split_folder)
                    
                    if result:
                        print("\n✅ 合并成功!")
                        print(f"📁 输出文件: {result}")
                    else:
                        print("\n❌ 合并失败，请查看日志了解详情")
                        
                elif choice == '4':
                    # 批量合并文件夹
                    folder_path = self.get_user_input("文件夹路径", self.path_validator)
                    if not folder_path:
                        continue
                    
                    folder_path = Path(folder_path)
                    print("🚀 开始批量合并...")
                    stats = self.batch_processor.batch_merge_images(folder_path)
                    
                    print("\n📊 批量合并结果:")
                    print(f"📁 处理文件夹: {folder_path}")
                    print(f"✅ 成功合并: {stats.successful}")
                    print(f"❌ 失败合并: {stats.failed}")
                    print(f"📈 成功率: {stats.success_rate:.1f}%")
                    print(f"⏱️ 总用时: {stats.elapsed_time:.2f}s")
                    
                elif choice == '5':
                    # 配置设置
                    self.show_config_menu()
                    
                elif choice == '6':
                    # 查看日志
                    self.show_log()
                    
                else:
                    print("❌ 无效选择，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n\n👋 用户中断操作")
                break
            except Exception as e:
                logger.error(f"程序运行异常: {e}")
                print(f"❌ 程序异常: {e}")
                print("💡 请查看日志文件获取详细信息")

def create_command_line_interface():
    """创建命令行接口"""
    parser = argparse.ArgumentParser(
        description="图片拆分与合并工具 v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --split image.png config.xml              # 拆分单个图片
  %(prog)s --batch-split /path/to/folder            # 批量拆分文件夹
  %(prog)s --merge image.png config.xml split_dir   # 合并单个图片
  %(prog)s --batch-merge /path/to/folder            # 批量合并文件夹
  %(prog)s --interactive                             # 启动交互模式
        """
    )
    
    # 操作模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--split', nargs=2, metavar=('IMAGE', 'XML'),
                           help='拆分单个图片 (图片文件 XML文件)')
    mode_group.add_argument('--batch-split', metavar='FOLDER',
                           help='批量拆分文件夹中的所有图片')
    mode_group.add_argument('--merge', nargs=3, metavar=('IMAGE', 'XML', 'SPLIT_DIR'),
                           help='合并单个图片 (原始图片 XML文件 拆分目录)')
    mode_group.add_argument('--batch-merge', metavar='FOLDER',
                           help='批量合并文件夹中的所有图片')
    mode_group.add_argument('--interactive', action='store_true',
                           help='启动交互式界面')
    
    # 配置选项
    parser.add_argument('--workers', type=int, default=None,
                       help='线程数 (默认: CPU核心数)')
    parser.add_argument('--format', choices=['PNG', 'JPEG', 'WEBP'],
                       default='PNG', help='输出格式 (默认: PNG)')
    parser.add_argument('--no-validate', action='store_true',
                       help='禁用尺寸验证')
    parser.add_argument('--no-fix-xml', action='store_true',
                       help='禁用XML自动修复')
    parser.add_argument('--config', default='config.json',
                       help='配置文件路径 (默认: config.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安静模式')
    
    return parser

def main():
    """主函数"""
    parser = create_command_line_interface()
    args = parser.parse_args()
    
    # 配置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 初始化配置
        config = ConfigManager(args.config)
        
        # 应用命令行参数
        if args.workers is not None:
            config.set('max_workers', args.workers)
        config.set('output_format', args.format)
        config.set('validate_dimensions', not args.no_validate)
        config.set('auto_fix_xml', not args.no_fix_xml)
        
        # 交互模式
        if args.interactive:
            ui = InteractiveUI()
            ui.run()
            return
        
        # 命令行模式
        batch_processor = BatchProcessor(config)
        image_processor = ImageProcessor(config)
        
        if args.split:
            # 单个文件拆分
            img_path = Path(args.split[0])
            xml_path = Path(args.split[1])
            
            if not img_path.exists():
                print(f"❌ 图片文件不存在: {img_path}")
                sys.exit(1)
            if not xml_path.exists():
                print(f"❌ XML文件不存在: {xml_path}")
                sys.exit(1)
            
            output_dir = img_path.parent / f"{img_path.stem}_split"
            stats = image_processor.split_image(img_path, xml_path, output_dir)
            
            print(f"拆分完成: 成功 {stats.successful}/{stats.total}, "
                  f"用时 {stats.elapsed_time:.2f}s")
            
        elif args.batch_split:
            # 批量拆分
            folder_path = Path(args.batch_split)
            
            if not folder_path.exists() or not folder_path.is_dir():
                print(f"❌ 文件夹不存在: {folder_path}")
                sys.exit(1)
            
            stats = batch_processor.batch_split_images(folder_path)
            print(f"批量拆分完成: 成功 {stats.successful}, "
                  f"失败 {stats.failed}, 用时 {stats.elapsed_time:.2f}s")
            
        elif args.merge:
            # 单个文件合并
            img_path = Path(args.merge[0])
            xml_path = Path(args.merge[1])
            split_dir = Path(args.merge[2])
            
            if not img_path.exists():
                print(f"❌ 图片文件不存在: {img_path}")
                sys.exit(1)
            if not xml_path.exists():
                print(f"❌ XML文件不存在: {xml_path}")
                sys.exit(1)
            if not split_dir.exists() or not split_dir.is_dir():
                print(f"❌ 拆分目录不存在: {split_dir}")
                sys.exit(1)
            
            result = image_processor.merge_images(img_path, xml_path, split_dir)
            if result:
                print(f"合并完成: {result}")
            else:
                print("❌ 合并失败")
                sys.exit(1)
                
        elif args.batch_merge:
            # 批量合并
            folder_path = Path(args.batch_merge)
            
            if not folder_path.exists() or not folder_path.is_dir():
                print(f"❌ 文件夹不存在: {folder_path}")
                sys.exit(1)
            
            stats = batch_processor.batch_merge_images(folder_path)
            print(f"批量合并完成: 成功 {stats.successful}/{stats.total}, "
                  f"用时 {stats.elapsed_time:.2f}s")
        
        # 保存配置
        config.save_config()
        
    except KeyboardInterrupt:
        print("\n👋 用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"程序异常: {e}")
        print(f"❌ 程序异常: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()