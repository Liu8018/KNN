#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <limits>

void readLabelFeatureList(std::string filePath, 
         std::vector<std::pair<std::string,std::vector<float>>> &labelFeatureMap)
{
    std::ifstream ifs(filePath,std::ios::in);
    if(!ifs.is_open()){
        std::cout<<"文件\""<<filePath<<"\"打开失败!"<<std::endl;
        exit(1);
    }
    
    std::string line;
    while(getline(ifs,line)){
        //去除换行符
        if(line[line.length()-1] == '\n'){
            line = line.substr(0,line.length()-1);
        }
        
        //找到分隔符
        int colonPos = line.find_first_of(":");
        if(colonPos == std::string::npos){
            std::cout<<"文件\""<<filePath<<"\"的内容格式不正确!"<<std::endl;
            exit(1);
        }
        
        //分隔符前为target
        std::string label = line.substr(0,colonPos);
        
        //分隔符后为feature
        std::string featureStr = line.substr(colonPos+1,line.length());
        std::vector<float> feature;
        while(1){
            int commaPos = featureStr.find_first_of(",");
            if(commaPos == std::string::npos){
                break;
            }
            float f = atof(featureStr.substr(0,commaPos).data());
            feature.push_back(f);
            
            featureStr = featureStr.substr(commaPos+1,featureStr.length());
        }
        
        labelFeatureMap.push_back(std::make_pair(label,feature));
    }
    
    ifs.close();
}

float norm(const std::vector<float> &vec)
{
    float v = 0;
    for(int i=0;i<vec.size();i++){
        v += vec[i]*vec[i];
    }
    v = std::sqrt(v);
    
    return v;
}

float getEuclideanDistance(const std::vector<float> &vec1,
                            const std::vector<float> &vec2)
{
    std::vector<float> diffVec(vec1.size());
    for(int i=0;i<vec1.size();i++){
        diffVec[i] = vec1[i] - vec2[i];
    }
    
    return norm(diffVec);
}

float getCosineDistance(const std::vector<float> &vec1,
                         const std::vector<float> &vec2)
{
    float mulv = 0;
    for(int i=0;i<vec1.size();i++){
        mulv += vec1[i] * vec2[i];
    }
    
    float cos = mulv/(norm(vec1)*norm(vec2));
    
    cos *= 0.5;
    cos += 0.5;
    cos = 1/cos;
    
    return cos;
}

float knnTest(std::string trainDataFile, std::string valDataFile)
{
    std::vector<std::pair<std::string,std::vector<float>>> trainData;
    std::vector<std::pair<std::string,std::vector<float>>> valData;
    readLabelFeatureList(trainDataFile,trainData);
    readLabelFeatureList(valDataFile,valData);
    
    int nsamples = valData.size();
    int ncorrect = 0;
    
    for(int vid=0;vid<valData.size();vid++){
        std::string trueLabel = valData[vid].first;
        std::string predLabel = "";
        float minDistance = std::numeric_limits<float>::max();
        
        for(int tid=0;tid<trainData.size();tid++){
            float distance = getCosineDistance(valData[vid].second,trainData[tid].second);
            if(distance < minDistance){
                predLabel = trainData[tid].first;
                minDistance = distance;
            }
        }
        
        if(trueLabel == predLabel){
            ncorrect++;
        }
    }
    
    float score = ncorrect/float(nsamples);
    std::cout<<"score:"<<score<<std::endl;
    
    return score;
}
