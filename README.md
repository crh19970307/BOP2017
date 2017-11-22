# BOP2017
-----
## Summary
This is our code for Beauty Of Programming 2017 Competition.

Our main contribution:
  * Propose a sequence to sequence approach to judge the Matching Degree of question-answer pair.
  * Propose a algorithm to answer a question based on seqtoseq approach and information distilling technique.
  * Develop a bot service based on Microsoft Azure for answering questions of everything about SJTU.
  * The project rank **13/1198** in Microsoft Beauty of Programming 2017.
----
## Explanation

Folder *data* contains train and validation dataset for question-answer pair, as well as data we get from web.

Folder *model* contains model of the LSTM trained on the train dataset and a similar implementation for LSTM in keras. 

Folder *script* contains the script for training, testing.

----
If you have questions or ideas, just create an issue. Ackonwledegment for my teammates [Weichao Mao](https://github.com/xizeroplus), [Zhiyuan Tang](https://github.com/silencender) and [Xiang Song]() .
