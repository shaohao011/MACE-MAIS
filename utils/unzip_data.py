import os
import zipfile

def unzip_all_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否是.zip文件
            if file.endswith('.zip'):
                zip_file_path = os.path.join(root, file)
                
                # 检查是否是有效的.zip文件
                if not zipfile.is_zipfile(zip_file_path):
                    print(f"文件不是有效的.zip文件: {zip_file_path}")
                    continue  # 跳过无效文件
                
                # 创建解压目录
                extract_folder = os.path.splitext(zip_file_path)[0]
                os.makedirs(extract_folder, exist_ok=True)
                
                # 解压文件
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder)
                    print(f"解压完成: {zip_file_path} -> {extract_folder}")
                except zipfile.BadZipFile:
                    print(f"文件损坏或不是有效的.zip文件: {zip_file_path}")
                except Exception as e:
                    print(f"解压失败: {zip_file_path}, 错误: {e}")

# 使用示例

folder_path = 'data/gulou/images'  # 替换为你的文件夹路径
unzip_all_in_folder(folder_path)