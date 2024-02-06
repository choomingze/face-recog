function  outputID=FaceRecognitionNew(trainImgSet, trainPersonID, testPath)
%% imporat VGGFace Network
vgg = importCaffeNetwork('.\VGG_FACE_deploy.prototxt','.\VGG_FACE.caffemodel');

%% Setting Up the network
l_graph = layerGraph(vgg);
rm_l_graph = removeLayers(l_graph, {'ClassificationOutput','prob','fc8'});
dl_net = dlnetwork(rm_l_graph);

%% Initializing the dimensional space for feature vector
trainTmpSet=zeros(4096,size(trainImgSet,4)); 

%% For loop to extract the fecture vector from the training data set
for i=1:size(trainImgSet,4)
    tmpI= trainImgSet(:,:,:,i);
    feat_vec = fv(tmpI,dl_net);
    trainTmpSet(:,i) = extractdata(feat_vec);    
end

outputID=[];
distance=[];

%% For loop to extract the fecture vector from the testing data set
testImgNames=dir([testPath,'*.jpg']);
for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:).name]);%load one of the test images
    feat_vec = fv(testImg,dl_net);
    Y = extractdata(feat_vec);
    
    for j=1:size(trainTmpSet,2)  % comparing the distance of training and testing feature vector      
        distance(j) = pdist([Y';trainTmpSet(:,j)'],'cosine');
%         distance(j) = pdist([Y';trainTmpSet(:,j)']);
    end
    [~,index] = min(distance);
    outputID = [outputID; trainPersonID(index,:)];
end    
end

%%  funtion to resize the image and save it in deep learning array
function feat_vec = fv(testImg,dl_net)
    tmpII = imresize(testImg,[224 224]);
    dlII = dlarray(double(tmpII),'SSCB');
    feat_vec = predict(dl_net,dlII);
end
