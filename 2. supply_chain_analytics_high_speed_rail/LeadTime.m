function [RouteTimePool1]=...
    LeadTime(DurationStaTraMatrix,ColumnPool,...
    FlowPlantWarehouseMatrix,TimePlaWarMatrix,...
    FlowWarehouseWarehouseMatrix,TimeWarWarMatrix,...
    FlowWarehouseStationMatrix,TimeWarStaMatrix,...
    FlowStationTrainMatrix,TimeStaTraMatrix,...
    Alpha,Belta,CapacityFacility)
[numberRoute,~]=size(ColumnPool);

for r=1:numberRoute
   routeTime=0;
%================supplier to warehouse===============
   Flow= FlowPlantWarehouseMatrix(ColumnPool(r,4),ColumnPool(r,3));
   Time=TimePlaWarMatrix(ColumnPool(r,4),ColumnPool(r,3));
   [T,~]=BPRDev(Flow,Time,Alpha(4),Belta(4),CapacityFacility(4));
   routeTime=routeTime+T;
   %=============²warehouse===========================
   Flow= FlowWarehouseWarehouseMatrix(ColumnPool(r,3),ColumnPool(r,3));
   Time=TimeWarWarMatrix(ColumnPool(r,3),ColumnPool(r,3));
   [T,~]=BPRDev(Flow,Time,Alpha(3),Belta(3),CapacityFacility(3));
   routeTime=routeTime+T;
   %=============warehouse to station===========================
   Flow= FlowWarehouseStationMatrix(ColumnPool(r,3),ColumnPool(r,2));
   Time=TimeWarStaMatrix(ColumnPool(r,3),ColumnPool(r,2));
   [T,~]=BPRDev(Flow,Time,Alpha(2),Belta(2),CapacityFacility(2));
   routeTime=routeTime+T;
   %=============station to train===========================
   Flow= FlowStationTrainMatrix(ColumnPool(r,1),ColumnPool(r,2));
   Time=TimeStaTraMatrix(ColumnPool(r,1),ColumnPool(r,2));
   [T,~]=BPRDev(Flow,Time,Alpha(1),Belta(1),CapacityFacility(1));
   routeTime=routeTime+T;
   RouteTimePool(r)=routeTime;
end
%=============Calculate lead time=================

RouteTimePool1=zeros(1,numberRoute);
for r=1:numberRoute
    MM=DurationStaTraMatrix(ColumnPool(r,1),ColumnPool(r,2));
    RouteTimePool1(r)=RouteTimePool(r)+MM;
end

end