function [MoneyShortage,MoneySurplus,PJDemandTrain]=FlowIncome(Route,DecayRoute,PurchasePrice,SellingPrice,WastePrice,FlowRoute,TrainDemand,EatRate)
[numberRoute,~]=size(Route);
[numberTrain,~]=size(TrainDemand);
PJDemandRoute=0*ones(1,numberRoute);
PJDemandTrain=0*ones(1,numberTrain);
for r=1:numberRoute
    for t=1:numberTrain
        if Route(r,1)==t
            PJDemandTrain(t)=PJDemandTrain(t)+FlowRoute(r)*DecayRoute(r);  
        end
    end
end
MoneySurplus=0;

for t=1:numberTrain
    MoneySurplus1=0;
    for i=1:1:ceil(PJDemandTrain(t))
        MoneySurplus1=MoneySurplus1+((SellingPrice-PurchasePrice)*i+(WastePrice-PurchasePrice)*(ceil(PJDemandTrain(t))-i))*binopdf(i,ceil(TrainDemand(t)),EatRate);
        %MoneySurplus1=MoneySurplus1+((SellingPrice-PurchasePrice)*i+(WastePrice-PurchasePrice)*(ceil(PJDemandTrain(t))-i))*poisspdf(i,ceil(TrainDemand(t))*EatRate);
        %MoneySurplus1=MoneySurplus1+[(SellingPrice-PurchasePrice)*i+(WastePrice-PurchasePrice)*(ceil(PJDemandTrain(t))-i)]*unifpdf(i,0,ceil(TrainDemand(t)));
    end
    MoneySurplus=MoneySurplus+MoneySurplus1;
end
MoneyShortage=0;
for t=1:numberTrain
    MoneyShortage1=0;
    for i=ceil(PJDemandTrain(t))+1:1:ceil(TrainDemand(t))
       MoneyShortage1=MoneyShortage1+(SellingPrice-PurchasePrice)*ceil(PJDemandTrain(t))*binopdf(i,ceil(TrainDemand(t)),EatRate);
       %MoneyShortage1=MoneyShortage1+(SellingPrice-PurchasePrice)*ceil(PJDemandTrain(t))*poisspdf(i,ceil(TrainDemand(t))*EatRate);
       %MoneyShortage1=MoneyShortage1+(SellingPrice-PurchasePrice)*ceil(PJDemandTrain(t))*unifpdf(i,0,ceil(TrainDemand(t)));
    end
    MoneyShortage=MoneyShortage+MoneyShortage1;
end

end