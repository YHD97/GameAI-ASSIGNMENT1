package players.GoupX;

import core.GameState;
import objects.Bomb;
import objects.GameObject;
import utils.Types;
import utils.Types.DIRECTIONS;
import utils.Types.TILETYPE;
import utils.Vector2d;

import java.util.*;

import static java.lang.Math.*;
import static java.lang.Math.min;
import static utils.Types.DIRECTIONS.values;
import static utils.Utils.*;
import static utils.Utils.positionIsPassable;

public class A_StarHeuristic extends StateHeuristic {

    private BoardStats rootBoardStats;
    private Random random;


    public A_StarHeuristic(GameState root, Random random) {
        this.random = random;
        rootBoardStats = new BoardStats(root, this.random);

    }

    @Override
    public double evaluateState(GameState gs) {
        boolean gameOver = gs.isTerminal();
        Types.RESULT win = gs.winner();

        // Compute a score relative to the root's state.
        BoardStats lastBoardState = new BoardStats(gs, this.random);
        double rawScore = rootBoardStats.score(lastBoardState);

        // TODO: Should we reserve -1 and 1 to LOSS and WIN, and shrink rawScore to be in [-0.5, 0.5]?
        // rawScore is in [-1, 1], move it to [-0.5, 0.5]
        rawScore /= 2.0;

        if(gameOver && win == Types.RESULT.LOSS)
            rawScore = -1;

        if(gameOver && win == Types.RESULT.WIN)
            rawScore = 1;

        return rawScore;
    }

    public static class BoardStats
    {
        //A*
        public static final int STEP = 10;

        private ArrayList<Node> openList = new ArrayList<Node>();
        private ArrayList<Node> closeList = new ArrayList<Node>();


        // 原来的文件
        int tick, nTeammates, nEnemies, blastStrength;
        boolean canKick;
        int nWoods;

        static double maxWoods = -1;
        static double maxBlastStrength = 10;

        // 0.4
        double FACTOR_SAFE_DIRECTIONS = 0.2;
        double FACTOR_BOMB_DIRECTIONS = 0.2;

        // 0.3
        double FACTOR_ENEMY;
        double FACTOR_TEAM;

        // 0.1
        double FACTOR_ENEMY_DIST = 0.1;

        // 0.2
        double FACTOR_CANKICK = 0.05;
        double FACTOR_BLAST = 0.05;
        //double FACTOR_ADJ_ENEMY = 0.12;
        double FACTOR_NEAREST_POWERUP = 0.05;
        double FACTOR_WOODS = 0.05;

        // State information
        private Random random;

        private Vector2d myPosition;
        private Types.TILETYPE[][] board;
        private ArrayList<Bomb> bombs;
        private ArrayList<GameObject> enemies;

        private HashMap<Types.TILETYPE, ArrayList<Vector2d>> items;
        private HashMap<Vector2d, Integer> dist;
        private HashMap<Vector2d, Vector2d> prev;

        // Extra state information (to be used as heuristics):

        // Directions in range of a bomb
        private HashMap<DIRECTIONS, Integer> directionsInRangeOfBomb = null;
        private Integer n_directionsInRangeOfBomb = null;

        // Safe directions
        private ArrayList<DIRECTIONS> safeDirections = null;
        private Integer n_safeDirections = null;

        // Adjacency to an enemy
        private Integer isAdjacentEnemy = null;

        // Distance to nearest enemy
        private Integer distanceToNearestEnemy = null;

        // Distance to nearest power-up, up to 10 (default: 1000 as max distance)
        private Integer distanceToNearestPowerUp = null;

        BoardStats(GameState gs, Random random) {

            this.random = random;

            nEnemies = gs.getAliveEnemyIDs().size();

            // Init weights based on game mode
            if (gs.getGameMode() == Types.GAME_MODE.FFA) {
                FACTOR_TEAM = 0;
                FACTOR_ENEMY = 0.3;
            } else {
                FACTOR_TEAM = 0.1;
                FACTOR_ENEMY = 0.2;
                nTeammates = gs.getAliveTeammateIDs().size();  // We only need to know the alive teammates in team modes
                nEnemies -= 1;  // In team modes there's an extra Dummy agent added that we don't need to care about
            }

            // Save game state information
            this.tick = gs.getTick();
            this.blastStrength = gs.getBlastStrength();
            this.canKick = gs.canKick();

            // Count the number of wood walls
            this.nWoods = 1;
            for (Types.TILETYPE[] gameObjectsTypes : gs.getBoard()) {
                for (Types.TILETYPE gameObjectType : gameObjectsTypes) {
                    if (gameObjectType == Types.TILETYPE.WOOD)
                        nWoods++;
                }
            }
            if (maxWoods == -1) {
                maxWoods = nWoods;
            }

            this.myPosition = gs.getPosition();
            this.board = gs.getBoard();
            int[][] bombBlastStrength = gs.getBombBlastStrength();
            int[][] bombLife = gs.getBombLife();
            int ammo = gs.getAmmo();
            int blastStrength = gs.getBlastStrength();
            ArrayList<Types.TILETYPE> enemyIDs = gs.getAliveEnemyIDs();
            int boardSizeX = board.length;
            int boardSizeY = board[0].length;

            this.bombs = new ArrayList<>();
            this.enemies = new ArrayList<>();

            for (int x = 0; x < boardSizeX; x++) {
                for (int y = 0; y < boardSizeY; y++) {

                    if(board[y][x] == Types.TILETYPE.BOMB){
                        // Create a bomb object
                        Bomb bomb = new Bomb();
                        bomb.setPosition(new Vector2d(x, y));
                        bomb.setBlastStrength(bombBlastStrength[y][x]);
                        bomb.setLife(bombLife[y][x]);
                        bombs.add(bomb);
                    }
                    else if(Types.TILETYPE.getAgentTypes().contains(board[y][x]) &&
                            board[y][x].getKey() != gs.getPlayerId()){ // May be an enemy
                        if(enemyIDs.contains(board[y][x])) { // Is enemy
                            // Create enemy object
                            GameObject enemy = new GameObject(board[y][x]);
                            enemy.setPosition(new Vector2d(x, y));
                            enemies.add(enemy); // no copy needed
                        }
                    }
                }
            }

            Container from_dijkstra = dijkstra(board, myPosition, bombs, enemies, 10);
            this.items = from_dijkstra.items;
            this.dist = from_dijkstra.dist;
            this.prev = from_dijkstra.prev;
        }

        /**
         * Computes score for a game, in relation to the initial state at the root.
         * Minimizes number of opponents in the game and number of wood walls. Maximizes blast strength and
         * number of teammates, wants to kick.
         * @param futureState the stats of the board at the end of the rollout.
         * @return a score [0, 1]
         */
        double score(BoardStats futureState)
        {
            int diffSafeDirections = futureState.getNumberOfSafeDirections() - this.getNumberOfSafeDirections();
            int diffDirectionsInRangeOfBomb = -(futureState.getNumberOfDirectionsInRangeOfBomb() - this.getNumberOfDirectionsInRangeOfBomb());

            int diffTeammates = futureState.nTeammates - this.nTeammates;
            int diffEnemies = -(futureState.nEnemies - this.nEnemies);

            int diffDistanceToNearestEnemy = -(futureState.getDistanceToNearestEnemy() - this.getDistanceToNearestEnemy());

            int diffWoods = -(futureState.nWoods - this.nWoods);
            int diffCanKick = futureState.canKick && !this.canKick ? 1 : 0;
            int diffBlastStrength = futureState.blastStrength - this.blastStrength;
            //int diffAdjacentEnemy = futureState.getIsAdjacentEnemy() - this.getIsAdjacentEnemy();
            int diffDistanceToNearestPowerUp = -(futureState.getDistanceToNearestPowerUp() - this.getDistanceToNearestPowerUp());

            return (diffSafeDirections / 4.0) * FACTOR_SAFE_DIRECTIONS
                    + (diffDirectionsInRangeOfBomb / 4.0) * FACTOR_BOMB_DIRECTIONS
                    + (diffEnemies / 3.0) * FACTOR_ENEMY
                    + diffTeammates * FACTOR_TEAM
                    + (diffDistanceToNearestEnemy / 10.0) * FACTOR_ENEMY_DIST
                    + (diffWoods / maxWoods) * FACTOR_WOODS
                    + diffCanKick * FACTOR_CANKICK
                    + (diffBlastStrength / maxBlastStrength) * FACTOR_BLAST
                    //+ diffAdjacentEnemy * FACTOR_ADJ_ENEMY
                    + (diffDistanceToNearestPowerUp / 10.0) * FACTOR_NEAREST_POWERUP;
        }

        //字典：方向：数字
        private HashMap<DIRECTIONS, Integer> getDirectionsInRangeOfBomb(){
            if(this.directionsInRangeOfBomb == null){
                this.directionsInRangeOfBomb = computeDirectionsInRangeOfBomb(this.myPosition, this.bombs, this.dist);
            }
            return this.directionsInRangeOfBomb;
        }

        //获取炸弹范围内的方向数
        private Integer getNumberOfDirectionsInRangeOfBomb(){
            if(this.n_directionsInRangeOfBomb == null){
                this.n_directionsInRangeOfBomb = getDirectionsInRangeOfBomb().size();
            }
            return this.n_directionsInRangeOfBomb;
        }

        //计算炸弹范围内的方向
        private HashMap<DIRECTIONS, Integer> computeDirectionsInRangeOfBomb(Vector2d myPosition, ArrayList<Bomb> bombs,
                                                                            HashMap<Vector2d, Integer> dist) {
            HashMap<DIRECTIONS, Integer> ret = new HashMap<>();

            for(Bomb bomb : bombs){
                Vector2d position = bomb.getPosition();

                if(!dist.containsKey(position))
                    continue;

                int distance = dist.get(position);
                int bombBlastStrength = bomb.getBlastStrength();

                if(distance > bombBlastStrength)
                    continue;

                if(myPosition == position){ // We are on a bomb. All directions are in range of bomb.
                    DIRECTIONS[] directions = values();

                    for (DIRECTIONS direction : directions) {
                        ret.put(direction, max(ret.getOrDefault(direction, 0), bombBlastStrength));
                    }
                }
                else if(myPosition.x == position.x){
                    if(myPosition.y < position.y){ // Bomb is right.
                        ret.put(DIRECTIONS.DOWN, max(ret.getOrDefault(DIRECTIONS.DOWN, 0), bombBlastStrength));
                    }
                    else{ // Bomb is left.
                        ret.put(DIRECTIONS.UP, max(ret.getOrDefault(DIRECTIONS.UP, 0), bombBlastStrength));
                    }
                }
                else if(myPosition.y == position.y){
                    if(myPosition.x < position.x){ // Bomb is down.
                        ret.put(DIRECTIONS.RIGHT, max(ret.getOrDefault(DIRECTIONS.RIGHT, 0), bombBlastStrength));
                    }
                    else{ // Bomb is up.
                        ret.put(DIRECTIONS.LEFT, max(ret.getOrDefault(DIRECTIONS.LEFT, 0), bombBlastStrength));
                    }
                }
            }
            return ret;
        }

        //获得安全的方向
        private ArrayList<DIRECTIONS> getSafeDirections(){
            if(this.safeDirections == null){
                this.safeDirections = computeSafeDirections(this.board, this.myPosition, getDirectionsInRangeOfBomb(),
                        this.bombs, this.enemies);
            }
            return this.safeDirections;
        }

        //获得安全方向的数量
        private Integer getNumberOfSafeDirections(){
            if(this.n_safeDirections == null){
                this.n_safeDirections = getSafeDirections().size();
            }
            return this.n_safeDirections;
        }

        //计算安全的方向
        private ArrayList<DIRECTIONS> computeSafeDirections(Types.TILETYPE[][] board, Vector2d myPosition,
                                                            HashMap<DIRECTIONS, Integer> unsafeDirections,
                                                            ArrayList<Bomb> bombs, ArrayList<GameObject> enemies) {
            // All directions are unsafe. Return a position that won't leave us locked.
            ArrayList<DIRECTIONS> safe = new ArrayList<>();

            if(unsafeDirections.size() == 4){

                Types.TILETYPE[][] nextBoard = new Types.TILETYPE[board.length][];
                for (int i = 0; i < board.length; i++) {
                    nextBoard[i] = new Types.TILETYPE[board[i].length];
                    for (int i1 = 0; i1 < board[i].length; i1++) {
                        if (board[i][i1] != null) {
                            // Power-ups array contains null elements, don't attempt to copy those.
                            nextBoard[i][i1] = board[i][i1];
                        }
                    }
                }

                nextBoard[myPosition.x][myPosition.y] = Types.TILETYPE.BOMB;

                for (Map.Entry<DIRECTIONS, Integer> entry : unsafeDirections.entrySet()){

                    DIRECTIONS direction = entry.getKey();
                    int bomb_range = entry.getValue();

                    Vector2d nextPosition = myPosition.copy();
                    nextPosition = nextPosition.add(direction.toVec());

                    if(!positionOnBoard(nextBoard, nextPosition) ||
                            !positionIsPassable(nextBoard, nextPosition, enemies))
                        continue;

                    if(!isStuckDirection(nextPosition, bomb_range, nextBoard, enemies)){
                        return new ArrayList<>(Arrays.asList(direction));
                    }
                }
                return safe;
            }

            // The directions that will go off the board.
            Set<DIRECTIONS> disallowed = new HashSet<>();

            DIRECTIONS[] directions = values();

            for (DIRECTIONS current_direction : directions) {

                Vector2d position = myPosition.copy();
                position = position.add(current_direction.toVec());

                DIRECTIONS direction = getDirection(myPosition, position);

                if(!positionOnBoard(board, position)){
                    disallowed.add(direction);
                    continue;
                }

                if(unsafeDirections.containsKey(direction)) continue;

                if(positionIsPassable(board, position, enemies) || positionIsFog(board, position)){
                    safe.add(direction);
                }
            }

            if(safe.isEmpty()){
                // We don't have any safe directions, so return something that is allowed.
                for(DIRECTIONS k : unsafeDirections.keySet()) {
                    if(!disallowed.contains(k))
                        safe.add(k);
                }
            }

            return safe;
        }

        private boolean isStuckDirection(Vector2d nextPosition, int bombRange, Types.TILETYPE[][] nextBoard,
                                         ArrayList<GameObject> enemies) {
            // A tuple class for PriorityQueue since it does not support pair of values in default
            class Tuple implements Comparable<Tuple>{
                private int distance;
                private Vector2d position;

                private Tuple(int distance, Vector2d position){
                    this.distance = distance;
                    this.position = position;
                }

                @Override
                public int compareTo(Tuple tuple) {
                    return this.distance - tuple.distance;
                }
            }

            PriorityQueue<Tuple> Q = new PriorityQueue<>();
            Q.add(new Tuple(0, nextPosition));

            Set<Vector2d> seen = new HashSet<>();

            boolean is_stuck = true;

            while(!Q.isEmpty()){
                Tuple tuple = Q.remove();
                int dist = tuple.distance;
                Vector2d position = tuple.position;

                seen.add(position);

                if(nextPosition.x != position.x && nextPosition.y != position.y){
                    is_stuck = false;
                    break;
                }

                if(dist > bombRange){
                    is_stuck = false;
                    break;
                }

                DIRECTIONS[] directions = values();

                for (DIRECTIONS direction : directions) {
                    Vector2d newPosition = position.copy();
                    newPosition = newPosition.add(direction.toVec());

                    if(seen.contains(newPosition)) continue;

                    if(!positionOnBoard(nextBoard, newPosition)) continue;

                    if(!positionIsPassable(nextBoard, newPosition, enemies)) continue;

                    dist = abs(direction.x() + position.x - nextPosition.x) +
                            abs(direction.y() + position.y - nextPosition.y);

                    Q.add(new Tuple(dist, newPosition));
                }
            }
            return is_stuck;
        }

        private int getIsAdjacentEnemy(){
            if(this.isAdjacentEnemy == null){
                this.isAdjacentEnemy = computeIsAdjacentEnemy(this.items, this.dist, this.enemies) ? 1 : 0;
            }
            return this.isAdjacentEnemy;
        }

        private boolean computeIsAdjacentEnemy(
                HashMap<Types.TILETYPE, ArrayList<Vector2d> > items,
                HashMap<Vector2d, Integer> dist,
                ArrayList<GameObject> enemies)
        {
            for(GameObject enemy : enemies){
                if(items.containsKey(enemy.getType())) {
                    ArrayList<Vector2d> items_list = items.get(enemy.getType());
                    for (Vector2d position : items_list) {
                        if (dist.get(position) == 1)
                            return true;
                    }
                }
            }
            return false;
        }

        //得到附近敌人的距离
        private int getDistanceToNearestEnemy(){
            if(this.distanceToNearestEnemy == null){
                this.distanceToNearestEnemy = computeDistanceToNearestEnemy(this.items, this.dist, this.enemies);
            }
            return this.distanceToNearestEnemy;
        }
        //Calculate the distance to the enemy
        private int computeDistanceToNearestEnemy(
                HashMap<Types.TILETYPE, ArrayList<Vector2d> > items,
                HashMap<Vector2d, Integer> dist,
                ArrayList<GameObject> enemies)
        {
            int distance = 1000; // TODO: Max distance/Infinity
            for(GameObject enemy : enemies){
                if(items.containsKey(enemy.getType())) {
                    ArrayList<Vector2d> items_list = items.get(enemy.getType());
                    for (Vector2d position : items_list) {
                        if(dist.get(position) < distance)
                            distance = dist.get(position);
                    }
                }
            }
            if(distance > 10)
                distance = 10;
            return distance;
        }

        //Get the distance to nearby PowerUp
        private int getDistanceToNearestPowerUp(){
            if(this.distanceToNearestPowerUp == null){
                this.distanceToNearestPowerUp = computeDistanceToNearestPowerUp(this.items);
            }
            return this.distanceToNearestPowerUp;
        }

        //Calculate the distance to nearby PowerUp
        private int computeDistanceToNearestPowerUp(HashMap<Types.TILETYPE, ArrayList<Vector2d> > items)
        {
            Vector2d previousNode = new Vector2d(-1, -1); // placeholder, these values are not actually used
            int distance = 1000; // TODO: Max distance/Infinity
            for (Map.Entry<Types.TILETYPE, ArrayList<Vector2d>> entry : items.entrySet()) {
                // check pickup entries on the board
                if (entry.getKey().equals(Types.TILETYPE.EXTRABOMB) ||
                        entry.getKey().equals(Types.TILETYPE.KICK) ||
                        entry.getKey().equals(Types.TILETYPE.INCRRANGE)){
                    // no need to store just get closest
                    for (Vector2d coords: entry.getValue()){
                        if (dist.get(coords) < distance){
                            distance = dist.get(coords);
                            previousNode = coords;
                        }
                    }
                }
            }
            if(distance > 10)
                distance = 10;
            return distance;
        }

        /**
         * Dijkstra's pathfinding
         * @param board - game board
         * @param myPosition - the position of agent
         * @param bombs - array of bombs in the game
         * @param enemies - array of enemies in the game
         * @param depth - depth of search (default: 10)
         * @return TODO
         */
        private Container dijkstra(Types.TILETYPE[][] board, Vector2d myPosition, ArrayList<Bomb> bombs,
                                   ArrayList<GameObject> enemies, int depth){

            HashMap<Types.TILETYPE, ArrayList<Vector2d> > items = new HashMap<>();
            HashMap<Vector2d, Integer> dist = new HashMap<>(); //目标到我的距离
            HashMap<Vector2d, Vector2d> prev = new HashMap<>(); // 上一个位置字典
            ArrayList<Vector2d> my_position = new ArrayList<>();

            //Data structures for the FIFO principle
            Queue<Vector2d> Q = new LinkedList<>();


            for(int r = max(0, myPosition.x - depth); r < min(board.length, myPosition.x + depth); r++){
                for(int c = max(0, myPosition.y - depth); c < min(board.length, myPosition.y + depth); c++){

                    Vector2d position = new Vector2d(r, c);

                    // Determines if two points are out of range of each other.
                    boolean out_of_range = (abs(c - myPosition.y) + abs(r - myPosition.x)) > depth;
                    if(out_of_range)
                        continue;

                    //Type of item returned
                    Types.TILETYPE itemType = board[r][c];
                    boolean positionInItems = (itemType == Types.TILETYPE.FOG ||
                            itemType == Types.TILETYPE.RIGID || itemType == Types.TILETYPE.FLAMES);
                    if(positionInItems)
                        continue;

                    if(itemType == TILETYPE.PASSAGE && !position.equals(myPosition)){
                        Q.add(position);
                    }
                    ArrayList<Vector2d> itemsTempList = items.get(itemType);
                    if(itemsTempList == null) {
                        //If the list is empty, create a new list of arrays
                        itemsTempList = new ArrayList<>();
                    }
                    itemsTempList.add(position);
                    //Add type and coordinates to the item dictionary
                    items.put(itemType, itemsTempList);

                    if(position.equals(myPosition)){
                        //If it's my location, add my location.
                       // my_position.add(position);
                        dist.put(position, 0);
                    }
                    else{
                        //Otherwise all distances are 100000
                        dist.put(position, 100000); // TODO: Inf
                    }
                }
            }

            //If the location of the bomb is equal to my location, 
            //add the bomb to the temporary items list and add it to the items dictionary.
            for(Bomb bomb : bombs){
                if(bomb.getPosition().equals(myPosition)){
                    ArrayList<Vector2d> itemsTempList = items.get(Types.TILETYPE.BOMB);
                    if(itemsTempList == null) {
                        itemsTempList = new ArrayList<>();
                    }
                    itemsTempList.add(myPosition);
                    items.put(Types.TILETYPE.BOMB, itemsTempList);
                }
            }
            //System.out.print("Q"+Q+"\n");
            //for(int i = 0; i<)

           if(dist.get(myPosition) != null){
               while (!Q.isEmpty()){
                   //end point
                   Vector2d position = Q.remove();
                   if(positionIsPassable(board, position, enemies)){

                       if(!dist.containsKey(position))
                           continue;

                       //start Position == Position
                       Node startNode = new Node(myPosition.x, myPosition.y);
                       Node endNode = new Node(position.x, position.y);
                       // The end point is returned, but by this time the parent node is already established and can be traced back to the start node

                       Node parent = findPath(startNode, endNode);
                       ArrayList<Node> arrayList = new ArrayList<Node>();


                       while (parent != null) {
                           // Iterate over the path just found。
                           //System.out.println(parent.x + ", " + parent.y);
                           arrayList.add(new Node(parent.x, parent.y));


                           //get currentPoint
                           Vector2d currentPoint = new Vector2d(parent.x, parent.y);

                           //get current dist

                           //现在储存的距离

                           if(dist.get(currentPoint) != null){
                               int currentDist = dist.get(currentPoint);
                               if(parent.parent != null){
                                   Vector2d LastPoint = new Vector2d(parent.parent.x, parent.parent.y);
                                   if(parent.F< currentDist){
                                       dist.put(currentPoint,parent.F);
                                       prev.put(currentPoint,LastPoint);

                                   }
                                   else if(currentDist==parent.F && random.nextFloat() < 0.5){
                                       dist.put(currentPoint,parent.F);
                                       prev.put(currentPoint,LastPoint);
                                   }

                               }
                           }


                           parent = parent.parent;
//                           System.out.print("distdddd2"+parent.parent.x+parent.parent.y+"\n");
//                           System.out.print("prevdddd2"+prev+"\n");


                       }




                   }










               }
           }

//           System.out.print("dist"+dist+"\n");
//           System.out.print("prev"+prev+"\n");


//            while (!Q.isEmpty()){
//                Vector2d position = Q.remove();
//                if(positionIsPassable(board, position, enemies)){
//                    DIRECTIONSNew[] directionsToBeChecked = DIRECTIONSNew.values();
//                    for (DIRECTIONSNew directionToBeChecked : directionsToBeChecked){
//                        Vector2d direction = directionToBeChecked.toVec();
//                        //End point
//                        Vector2d new_position = new Vector2d(position.x + direction.x, position.y + direction.y);
//
//                        if(!dist.containsKey(new_position))
//                            continue;
//                        Node startNode = new Node(position.x, position.y);
//                        Node endNode = new Node(new_position.x, new_position.y);
//                // The end point is returned, but by this time the parent node is already established and can be traced back to the start node
//
//                        Node parent = findPath(startNode, endNode);
//                        ArrayList<Node> arrayList = new ArrayList<Node>();
//
//
//                        while (parent != null) {// Iterate over the path just found。
//                            //System.out.println(parent.x + ", " + parent.y);
//                            arrayList.add(new Node(parent.x, parent.y));
//                            dist.put(new_position,parent.F);
//                            prev.put(new_position,position);
//                            //Q.add(new_position);
//                            Vector2d currentPoint = new Vector2d(parent.x, parent.y);
//                            int currentDist = dist.get(currentPoint);
//                            if(currentDist<parent.F){
//                                dist.put(new_position,parent.F);
//                                prev.put(new_position,position);
//                                Q.add(new_position);
//                            }
//                            else if(currentDist==parent.F && random.nextFloat() < 0.5){
//                                dist.put(new_position,parent.F);
//                                prev.put(new_position,position);
//                            }
//                            parent = parent.parent;
//
//
//
//                        }
//                        //Q.add(new_position);
////                        System.out.print("Q"+Q+"\n");
////                        System.out.print("dist"+dist+"\n");
////                        System.out.print("prev"+prev+"\n");
//
//
//
//                    }
//
//
//
//                }
//
//            }






            Container container = new Container();
            //
            container.dist = dist;
            //Coordinates of all items items.put()
            container.items = items;
            //Return to dictionary {last position: previous position}
            container.prev = prev;

            return container;
        }

        // Container for return values of Dijkstra's pathfinding algorithm.
        private class Container {
            HashMap<Types.TILETYPE, ArrayList<Vector2d> > items;
            HashMap<Vector2d, Integer> dist;
            HashMap<Vector2d, Vector2d> prev;
            Container() { }
        }


        //a*
        public enum DIRECTIONSNew { //方向
            LEFT(-1, 0),
            RIGHT(1, 0),
            UP(0, -1),
            DOWN(0, 1);

            private int x, y;

            DIRECTIONSNew(int x, int y) { // 坐标
                this.x = x;
                this.y = y;
            }

            public Vector2d toVec() { // 二维矢量
                return new Vector2d(x, y);
            }

            public int x() {return x;}
            public int y() {return y;}
        }




        // A* 节点
        public Node findMinFNodeInOpenList() {
             // Start with the F of the first element as the minimum value, 
             //then iterate through all the values of the openlist to find the minimum value
            Node tempNode = openList.get(0);
            for (Node node : openList) {
                if (node.F < tempNode.F) {
                    tempNode = node;
                }
            }
            return tempNode;
        }

        // When considering surrounding nodes, 
        //nodes with a node value of 1 are not taken into account, so naturally, obstacles are avoided directly
        public ArrayList<Node> findNeighborNodes(Node currentNode) {
            ArrayList<Node> arrayList = new ArrayList<Node>();
            // Only top, bottom, left and right are considered, not diagonal
            int topX = currentNode.x;
            int topY = currentNode.y - 1;
            // The canReach method ensures that the subscript is not out of bounds 
            //The exists method ensures that this adjacent node does not exist in the closeList, i.e. 
            //it has not been traversed before
            if (canReach(topX, topY) && !exists(closeList, topX, topY)) {
                arrayList.add(new Node(topX, topY));
            }
            int bottomX = currentNode.x;
            int bottomY = currentNode.y + 1;
            if (canReach(bottomX, bottomY) && !exists(closeList, bottomX, bottomY)) {
                arrayList.add(new Node(bottomX, bottomY));
            }
            int leftX = currentNode.x - 1;
            int leftY = currentNode.y;
            if (canReach(leftX, leftY) && !exists(closeList, leftX, leftY)) {
                arrayList.add(new Node(leftX, leftY));
            }
            int rightX = currentNode.x + 1;
            int rightY = currentNode.y;
            if (canReach(rightX, rightY) && !exists(closeList, rightX, rightY)) {
                arrayList.add(new Node(rightX, rightY));
            }
            return arrayList;
        }

        public boolean canReach(int x, int y) {
            if (x >= 0 && x < board.length && y >= 0 && y < board[0].length) {
                return board[x][y]== TILETYPE.PASSAGE; // 原来是在这里避过障碍物的啊。。如果节点值不为0，说明不可到达。
            }
            return false;
        }

        public Node findPath(Node startNode, Node endNode) {

            // Add the starting point to the open list
            openList.add(startNode);

            while (openList.size() > 0) {
                // Iterate through the open list to find the node 
                //with the smallest F value and use it as the current node to be processed
                Node currentNode = findMinFNodeInOpenList();

                // The node with the lowest F value is removed from the open list
                openList.remove(currentNode);
                // Move this node to the close list, the closelist is the chain that stores the paths
                closeList.add(currentNode);

                // Find surrounding nodes that do not exist in the close list (disregarding the neighbours of the hypotenuse)
                ArrayList<Node> neighborNodes = findNeighborNodes(currentNode);

                // The openlist is actually a collection of stored peripheral nodes
                for (Node node : neighborNodes) {// add the neighbour nodes to the openlist
                    if (exists(openList, node)) { // If a neighbour node is in the openlist
                        foundPoint(currentNode, node);
                    } else {
                        // If a neighbouring node is not in the openlist, then add it to the openlist
                        notFoundPoint(currentNode, endNode, node);
                    }
                }
   // If the end point is found in the openlist, then the path has been found and the end point is returned
   if (find(openList, endNode) != null) {
                    return find(openList, endNode);
                }
            }

            return find(openList, endNode);
        }

        // In this case, the node with the smallest F value has been traversed before, 
        //so the G,H,F values of this node have already been calculated.
        // At this point the H value will definitely not change, so we have to compare the G values, 
        //if the current G value is smaller than the previous one, it means that the current path is better
        // Then reset the parent pointer, G and F values of this node
        private void foundPoint(Node tempStart, Node node) {
            int G = calcG(tempStart, node);
            if (G < node.G) {
                node.parent = tempStart;
                node.G = G;
                node.calcF();
            }
        }

        //In this case, the value of this node has not been calculated before,
        //so here we have to calculate the G,H,F values again, then check the parent pointer and add it to the openlist
        private void notFoundPoint(Node tempStart, Node end, Node node) {
            node.parent = tempStart;
            node.G = calcG(tempStart, node);
            node.H = calcH(end, node);
            node.calcF();
            openList.add(node);
        }

        private int calcG(Node start, Node node) {
            int G = STEP;
            int parentG = node.parent != null ? node.parent.G : 0;
            return G + parentG;
        }

        // Calculating the H-value method
        private int calcH(Node end, Node node) {
            int step = Math.abs(node.x - end.x) + Math.abs(node.y - end.y);
            return step * STEP;
        }

        public static Node find(List<Node> nodes, Node point) {
            for (Node n : nodes)
                if ((n.x == point.x) && (n.y == point.y)) {
                    return n;
                }
            return null;
        }

        public static boolean exists(List<Node> nodes, Node node) {
            for (Node n : nodes) {
                if ((n.x == node.x) && (n.y == node.y)) {
                    return true;
                }
            }
            return false;
        }

        public static boolean exists(List<Node> nodes, int x, int y) {
            for (Node n : nodes) {
                if ((n.x == x) && (n.y == y)) {
                    return true;
                }
            }
            return false;
        }

        public static class Node {
            public Node(int x, int y) {
                this.x = x;
                this.y = y;
            }

            public int x;
            public int y;

            public int F;
            public int G;
            public int H;

            public void calcF() {
                this.F = this.G + this.H;
            }

            public Node parent;
        }

    }
}




