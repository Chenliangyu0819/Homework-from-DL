# - 主要内容：

- 1基于CNN的手写数字识别实验
        
        本文基于改版的 AlexNet 经典卷积神经网络针对手写数字识别问题进行实验分析。基础的 AlexNet 是 2012 年 ImageNet 竞赛冠军获得者 Hinton 和他的学生Alex Krizhevsky 设计的。本文将 AlexNet 进行一定程度的改进，将其适用于处理MNIST 数据集的训练和测试，模型最终在测试集上的准确率达到 98.6%。

- 2基于VGG16预训练模型的猫狗分类实验
        
        本文使用预训练模型 VGG16 框架，结合精调技术对猫狗分类任务进行实验。在个人电脑配置有限的情况下，有效利用了数据增强技术，并搭配预训练模型对猫狗图片实现精准分类，最终在测试集上的准确率达到92.0%。

- 3基于LSTM的自动写诗实验
        
        本次试验任务是基于唐诗数据集的文本预测工作，根据给定的若干汉字进行诗句续写，或者生成藏头诗。本文基于 LSTM 网络进行训练与测试，能够较好地实现自动写诗任务。

- 4基于TextCNN网络框架的电影评论文本分类实验
        
        情感分析（Sentiment Analysis）是自然语言处理领域的一个重要的研究方向。它的目的是挖掘文本要表达的情感观点，对文本按情感倾向进行分类。本次任务是针对中文电影评论的情感倾向进行分类，类别包括正向和负向情感。本文使用TextCNN 网络对文本数据进行分析，在大量实验和调参的基础上，模型最终在测试集上达到 87.26%的分类准确率。



# Project Title

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
