Описание файлов:  
  * `data/` - папка содержащая два тренировачных текстовых файла - один небольшой, второй побольше.
  * `data_reader.py` - класс, который отвечает за предпроцессинг текстовых файлов.
  * `mi_rnn_cell.py` - переписанные классы MIGRUCell, MILSTMCell и mi_linear метод.
  * `model_words_generation.py` - реализованная Character-Level сетка для сравнительных тестов.
  
Примеры запуска:  
  * `python model_words_generation.py --model=milstm --display_epoch=5 --num_epochs=100 --log_dir=logs_small`
  * `python model_words_generation.py --model=lstm --data_path=data/input_large.txt --num_epochs=200 --batch_size=200 --sequence_size=50 --display_epoch=1`
