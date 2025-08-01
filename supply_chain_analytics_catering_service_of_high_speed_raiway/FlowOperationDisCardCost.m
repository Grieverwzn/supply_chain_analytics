function[TotalOC,TotalDC]=FlowOperationDisCardCost(FlowRoute,Route,DecayPlaWarMatrix,VarCostPlaWarMatrix,FixCostPlaWarMatrix,DecayWarWarMatrix,VarCostWarWarMatrix,FixCostWarWarMatrix,DecayWarStaMatrix,VarCostWarStaMatrix,FixCostWarStaMatrix,DecayStaTraMatrix,VarCostStaTraMatrix,FixCostStaTraMatrix,Capacity,PurchasePrice,WastePrice)
[numberRoute,~]=size(Route);


[numberPlant,numberWarehouse]=size(DecayPlaWarMatrix);
[numberTrain,numberStation]=size(DecayStaTraMatrix);
FlowStationTrainMatrix=zeros(numberTrain,numberStation);
FlowWarehouseStationMatrix=zeros(numberWarehouse,numberStation);
FlowWarehouseWarehouseMatrix=zeros(numberWarehouse,numberWarehouse);
FlowPlantWarehouseMatrix=zeros(numberPlant,numberWarehouse);
for r=1:numberRoute
    FlowPlantWarehouseMatrix(Route(r,4),Route(r,3))=FlowPlantWarehouseMatrix(Route(r,4),Route(r,3))+FlowRoute(r);
    Decay=DecayPlaWarMatrix(Route(r,4),Route(r,3));
    FlowWarehouseWarehouseMatrix(Route(r,3),Route(r,3))=FlowWarehouseWarehouseMatrix(Route(r,3),Route(r,3))+FlowRoute(r)*Decay;
    Decay=Decay*DecayWarWarMatrix(Route(r,3),Route(r,3));
    FlowWarehouseStationMatrix(Route(r,3),Route(r,2))=FlowWarehouseStationMatrix(Route(r,3),Route(r,2))+FlowRoute(r)*Decay;
    Decay=Decay*DecayWarStaMatrix(Route(r,3),Route(r,2));
    FlowStationTrainMatrix(Route(r,1),Route(r,2))=FlowStationTrainMatrix(Route(r,1),Route(r,2))+FlowRoute(r)*Decay;
end

%-----------------different layers---------------
OCPlaWar=0;
DCPlaWar=0;
for i=1:numberPlant
    for j=1:numberWarehouse
    Flow=FlowPlantWarehouseMatrix(i,j);
    Decay=DecayPlaWarMatrix(i,j);
    OCPlaWar=OCPlaWar+(Flow*(VarCostPlaWarMatrix(i,j))+FixCostPlaWarMatrix(i,j)/Capacity(4))*Flow;
    DCPlaWar=DCPlaWar+(PurchasePrice-WastePrice)*(1-Decay)*Flow;
    end
end
OCWarWar=0;
DCWarWar=0;
for i=1:numberWarehouse
    for j=1:numberWarehouse
    Flow=FlowWarehouseWarehouseMatrix(i,j);
    Decay=DecayWarWarMatrix(i,j);
    OCWarWar=OCWarWar+(Flow*(VarCostWarWarMatrix(i,j))+FixCostWarWarMatrix(i,j)/Capacity(3))*Flow;
    DCWarWar=DCWarWar+(PurchasePrice-WastePrice)*(1-Decay)*Flow;
    end
end
OCWarSta=0;
DCWarSta=0;
for i=1:numberWarehouse
    for j=1:numberStation
    Flow=FlowWarehouseStationMatrix(i,j);
    Decay=DecayWarStaMatrix(i,j);
    OCWarSta=OCWarSta+(Flow*(VarCostWarStaMatrix(i,j))+FixCostWarStaMatrix(i,j)/Capacity(2))*Flow;
    DCWarSta=DCWarSta+(PurchasePrice-WastePrice)*(1-Decay)*Flow;
    end
end
OCStaTra=0;
DCStaTra=0;
for i=1:numberTrain
    for j=1:numberStation
    Flow=FlowStationTrainMatrix(i,j);
    Decay=DecayStaTraMatrix(i,j);
    OCStaTra=OCStaTra+(Flow*(VarCostStaTraMatrix(i,j))+FixCostStaTraMatrix(i,j)/Capacity(1))*Flow;
    DCStaTra=DCStaTra+(PurchasePrice-WastePrice)*(1-Decay)*Flow;
    end
end
%------------------------------------summation-------------------------------------
TotalOC=OCPlaWar+OCWarWar+OCWarSta+OCStaTra;
TotalDC=DCPlaWar+DCWarWar+DCWarSta+DCStaTra;

end