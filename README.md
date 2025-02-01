# YOLO 기반 점자블록 검출을 통한 시각장애인 이동 보조 장치
 본 프로젝트는 임베디드 시스템 환경에서 점자 블록을 검출하고 검출 결과에 대해서 길 정보를 인식하여 음성으로 출력하는 이동 보조 시스템입니다. 또한 임베디드 시스템의 제한된 자원에서 효율적인 추론 성능을 보이기 위해 TensorRT 최적화 엔진 적용하고 가중치의 정밀도를 축소하여 경량화합니다. 

## Published Paper
> <a href="https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11510186">**임베디드 기기를 위한 딥러닝 점자블록 인식 방법**</a>
>
> Author : 김희진, 윤재혁, 권순각
>
><p align="center">
    ><img src="https://github.com/user-attachments/assets/c7cb440c-4776-4445-82ab-c1211239f8d8" width="400px"/>
    ><br>
    ><sup> figure. Example of T-cross recognition</sup>
></p>

## Project Overview
>- **Project Name** : YOLO 기반 점자블록 검출을 통한 시각장애인 이동 보조 장치
>- **Project** : 2023.03 - 2023.8
>- **Project Language** : Python

## System Architecture
><p align="center">
    ><img src="https://github.com/user-attachments/assets/b6ee253a-1a24-43e0-9885-e317b72b5d96" width="600px"/>
    ><br>
    ><sup> figure. Example of T-cross recognition</sup>
></p>
>
>- **Embedded Device** : 임베디드 시스템은 NVIDIA사의 Jetson Nano 보드에서 개발 되었으며, 카메라는 Logitech Webcam C170를 사용하였습니다.
>- **Model for block detection** : YOLOv5 및 YOLOv8을 기반으로 개발되었으며, 임베디드 시스템의 한정된 자원에서 효율적인 프레임 처리 성능(FPS; Frame Per Second)을 보이기 위해 TensorRT 엔진을 적용합니다.
>
>>### 🛠 Tech Stack
>>| Category         | Tech Stack |
>>|-------------|-------------------------------------------------|
>>| **Embedded Device** | ![Python](https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white) |
>>| **Model Development**  | ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white) |

## Project goal
>- 15 FPS 이상으로 점자 블록 인식
>- 점자 블록 구성에 따른 경로 인식
>- 인식된 경로 음성 출력

## Project Demo
> <a href="https://www.youtube.com/watch?v=CzOFpBjnR4k">**딥러닝 시각장애인 이동 보조 시스템 데모 영상**</a>