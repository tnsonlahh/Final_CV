Giải thích file:
- helpers.py: có mấy hàm lấy embedding
- main và main2: chạy process local, trả ra ảnh opencv
- preprocessing_metadata: clean cái file meta gốc là fashion.csv
- upload_to_qdrant: up cái metadata kia kèm theo embedding của ảnh và title lên DB (cần chạy docker tạo qdrant đã)