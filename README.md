Giải thích file:
- helpers.py: có mấy hàm lấy embedding
- main: chạy process local, trả ra ảnh opencv
- main2: file ngu kệ nó đi 
- preprocessing_metadata: clean cái file meta gốc là fashion.csv
- upload_to_qdrant: up cái metadata kia kèm theo embedding của ảnh và title lên DB (cần chạy docker tạo qdrant đã cái này gen chat 1 dòng)
- .json: file kết quả khi test với 10 queries (cũng hơi tà đạo tí)
- evaluate: file tạo json kia kìa, mấy cái metric muốn hiểu thì gen chat file này chứ tôi lwofi viết tử tế lắm :))
gudluck