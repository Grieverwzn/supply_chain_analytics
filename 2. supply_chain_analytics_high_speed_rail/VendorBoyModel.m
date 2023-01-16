function [ProjectDemandRoute,ProjectDemandTrain,...
    TrainDemandRoute,VendorBoy]=...
    VendorBoyModel(EatRate,Route,DecayRoute,numberTrain,...
    numberRoute,FlowRoute,TrainDemand,SellingPrice,...
    PurchasePrice,WastePrice)

ProjectDemandRoute=0*ones(1,numberRoute);
ProjectDemandTrain=0*ones(1,numberTrain);
TrainDemandRoute=0*ones(1,numberRoute);

for r=1:numberRoute
    for t=1:numberTrain
        if Route(r,1)==t
            % calculate the project demand on each train for each iteration
            ProjectDemandTrain(t)=...
                ProjectDemandTrain(t)+FlowRoute(r)*DecayRoute(r);  
        end
    end
end
for r=1:numberRoute
    for t=1:numberTrain
        if Route(r,1)==t
            ProjectDemandRoute(r)=ProjectDemandTrain(t);  
            % project train's flow onto associated routes 
            TrainDemandRoute(r)=TrainDemand(t);
        end
    end
end 


%======================Derivative of vendor model================================
% based on binomial distribution
% trial number ceil(TrainDemandRoute(r))
% probability = EatRate
% ceil(ProjectDemandRoute(r)) numbers of passengers eating. 
VendorBoy=0*ones(1,numberRoute);
for r=1:numberRoute
aa= DecayRoute(r)*(SellingPrice-PurchasePrice);
bb= -DecayRoute(r)*(SellingPrice-WastePrice)*binocdf(ceil(ProjectDemandRoute(r)),ceil(TrainDemandRoute(r)),EatRate);
VendorBoy(r)= (aa+bb);
end
end