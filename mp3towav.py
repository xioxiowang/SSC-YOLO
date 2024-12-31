from pydub import AudioSegment
import os

def convert_mp3_to_wav(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            # 支持中文及大小写扩展名
            if filename.lower().endswith('.mp3'):
                try:
                    mp3_path = os.path.join(root, filename)
                    
                    # 保留子目录结构
                    relative_path = os.path.relpath(root, input_folder)
                    wav_folder = os.path.join(output_folder, relative_path)
                    os.makedirs(wav_folder, exist_ok=True)

                    # 替换扩展名
                    wav_path = os.path.join(wav_folder, filename.replace('.mp3', '.wav'))

                    # 读取和导出音频
                    audio = AudioSegment.from_file(mp3_path, format="mp3")
                    audio.export(wav_path, format='wav')
                    print(f"已转换: {mp3_path} -> {wav_path}")

                except Exception as e:
                    print(f"转换失败: {filename}, 错误: {e}")

    print("全部文件转换完成！")

# 输入输出文件夹
input_folder = "C:\\Users\\wangdada\\Desktop\\上班\\音频"
output_folder = "C:\\Users\\wangdada\\Desktop\\上班\\音频_wav"

convert_mp3_to_wav(input_folder, output_folder)
 