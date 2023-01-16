function [StoreTime,Timetable,Terminal_Timetable, DurationStaTraMatrix,...
    TimeStaTraMatrix,DecayStaTraMatrix,FixCostStaTraMatrix,...
    VarCostStaTraMatrix,DecayWare,FixCostWare,VarCostWare,...
    WarehouseStationMatrix,TimeWarStaMatrix,...
    DecayWarStaMatrix,FixCostWarStaMatrix,VarCostWarStaMatrix,...
    PlantWarehouseMatrix,TimePlaWarMatrix,DecayPlaWarMatrix,...
    FixCostPlaWarMatrix,VarCostPlaWarMatrix,OStationMatrix,DStationMatrix]...
    = reading(LSpan)


%% =============1. Read stations to trains matrix data=========================
station_to_train_matrix = csvread('input_station_to_train.csv',1);
indTrain = station_to_train_matrix(:,1); % train id 
indStation = station_to_train_matrix(:,2); % station id
FC = station_to_train_matrix(:,3); % fixed cost
VC = station_to_train_matrix(:,4); % variable cost 
TIME = station_to_train_matrix(:,5); % loading time at stations
DECAY = station_to_train_matrix(:,6); % decay rate of food products 
arrival_time = station_to_train_matrix(:,7); % arrival time at stations 
terminal_arrival_time = station_to_train_matrix(:,8); % arrival time at terminal stations 

% use sparse matrix representation 
Timetable=sparse(indTrain,indStation,arrival_time); % timetable represents the arrival time of different stations 
Terminal_Timetable = sparse(indTrain,indStation,terminal_arrival_time);
FixCostStaTraMatrix=sparse(indTrain,indStation,FC);
VarCostStaTraMatrix=sparse(indTrain,indStation,VC);
TimeStaTraMatrix=sparse(indTrain,indStation,TIME);
DecayStaTraMatrix=sparse(indTrain,indStation,DECAY);


% Define DurationStaTraMatrix 
% DurationStaTraMatrix represents the time food products staying in the train
% DurationStaTraMatrix(i,j) indicates the staying time of food products on
% train i if the products are loaded at station j till (the trains') terminal stations. 
[numberTrain,numberStation]=size(Timetable);
DurationStaTraMatrix=zeros(numberTrain,numberStation);
OStationMatrix=zeros(numberTrain,numberStation); % origin station  of a train 
DStationMatrix=zeros(numberTrain,numberStation); % destination station of a train 


for i=1:numberTrain
    for j=1:numberStation
       if Timetable(i,j)~=0
        %DurationStaTraMatrix(i,j)=max(Timetable(i,:))-Timetable(i,j); not correct
        % 1440 means the minutes of one whole day 
        % this is used to address the overnight trains: 
        % for example if a train from beijing has a departure time is 11:00 pm and
        % arive at shanghai at 6:00, passengers will stay 7 hours (420
        % minutes)in the train, i.e., mod (6*60 - 23*60,1440) = 420
            DurationStaTraMatrix(i,j)=mod(Terminal_Timetable(i,j)-Timetable(i,j),1440);
            if DurationStaTraMatrix(i,j) == 0 
                DStationMatrix(i,j) = 1; % end station =1
            end 
       end
    end
    [~,index] = max(DurationStaTraMatrix(i,:)); 
    OStationMatrix(i,index) = 1; % start station = 1
end


    
for i=1:numberTrain
    for j=1:numberStation
        if Timetable(i,j)==0
            DurationStaTraMatrix(i,j)=LSpan; % just a large number to stipulate the loading is impossible 
        end
    end
end    
%% =============2. Read warehouses data=========================
warehouse_matrix = csvread('input_warehouse.csv',1);
FixCostWare = warehouse_matrix(:,2);
VarCostWare = warehouse_matrix(:,3);
DecayWare = warehouse_matrix(:,4);
StoreTime=warehouse_matrix(:,5);          
DecayWare=DecayWare';
FixCostWare=FixCostWare';
VarCostWare=VarCostWare';


%==============3. Read warehouses to stations matrix data============
warehouse_to_station_matrix = csvread('input_warehouse_to_station.csv',1);
indWarehouse =  warehouse_to_station_matrix(:,1);
indStation = warehouse_to_station_matrix(:,2);
FC = warehouse_to_station_matrix(:,3);
VC = warehouse_to_station_matrix(:,4);
TIME = warehouse_to_station_matrix(:,5);
DECAY = warehouse_to_station_matrix(:,6);

[numberWarehouse,~]=size(indWarehouse);
WarehouseStationMatrix=sparse(indWarehouse,indStation,ones(numberWarehouse,1));
FixCostWarStaMatrix=sparse(indWarehouse,indStation,FC);
VarCostWarStaMatrix=sparse(indWarehouse,indStation,VC);
TimeWarStaMatrix=sparse(indWarehouse,indStation,TIME);
DecayWarStaMatrix=sparse(indWarehouse,indStation,DECAY);

%===============4. Read suppliers (plants) to warehouses matrix data==========
supplier_to_warehouse_matrix = csvread('input_supplier_to_warehouse.csv',1);
indPlant = supplier_to_warehouse_matrix(:,1);
indWarehouse = supplier_to_warehouse_matrix(:,2);
FC = supplier_to_warehouse_matrix(:,3);
VC = supplier_to_warehouse_matrix(:,4);
TIME = supplier_to_warehouse_matrix(:,5);
DECAY = supplier_to_warehouse_matrix(:,6);


[numberPlant,~]=size(indPlant);
PlantWarehouseMatrix=sparse(indPlant,indWarehouse,ones(numberPlant,1));
FixCostPlaWarMatrix=sparse(indPlant,indWarehouse,FC);
VarCostPlaWarMatrix=sparse(indPlant,indWarehouse,VC);
TimePlaWarMatrix=sparse(indPlant,indWarehouse,TIME);
DecayPlaWarMatrix=sparse(indPlant,indWarehouse,DECAY);

end

