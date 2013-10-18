package edu.umn.cs.recsys.uu;

import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.dao.ItemEventDAO;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.symbols.Symbol;
import org.grouplens.lenskit.symbols.TypedSymbol;
import org.grouplens.lenskit.vectors.ImmutableSparseVector;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.Set;

/**
 * User-user item scorer.
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SimpleUserUserItemScorer extends AbstractItemScorer {
    private final UserEventDAO userDao;
    private final ItemEventDAO itemDao;

    @Inject
    public SimpleUserUserItemScorer(UserEventDAO udao, ItemEventDAO idao) {
        userDao = udao;
        itemDao = idao;
    }

    @Override
    public void score(long user, @Nonnull MutableSparseVector scores) {
        SparseVector userVector = getUserRatingVector(user);

        // This is the loop structure to iterate over items to score
        for (VectorEntry e: scores.fast(VectorEntry.State.EITHER)) {
            // TODO Score items for this user using user-user collaborative filtering
            // Find the neighbors
            // a. Find the users who has rated the item
            long itemId = e.getKey(); //scores.keyDomain().firstLong();
            LongSet itemUsers = this.itemDao.getUsersForItem(itemId); // users who rated itemId
            MutableSparseVector userSims = MutableSparseVector.create(itemUsers);  // collection of itemUser:similarity
            userSims.fill(0);
            // b. Iterate thru the users
            for(long itemUserId:itemUsers){
                 // i. get the vector for the user in itemUsers
                SparseVector itemUserVector = getUserRatingVector(itemUserId);
                // ii. calculate the cosine between userVector and itemUserVector
                double similarity = CalcSimilarity(userVector, itemUserVector);
                userSims.set(itemUserId, similarity);
            }
            LongArrayList sortedUsers = userSims.keysByValue(true);
            // now, we can calculate the score for user
            // a. find the denominator
            double numerator = 0;
            double denominator = 0;
            int counter = 0; // count 30 neighbours
            int i = 0; // index
            while(counter<30){

                long neighborId = sortedUsers.get(i++);

                if(neighborId == user){
                    continue;
                }

                // s(u,v)
                double neighborSimilarity = userSims.get(neighborId);
                SparseVector neighborRatingVector = getUserRatingVector(neighborId);
                // r
                double neighborRating = neighborRatingVector.get(itemId);
                // mu
                double meanNeighborRating = neighborRatingVector.mean();

                numerator += neighborSimilarity * (neighborRating - meanNeighborRating);
                denominator += Math.abs(neighborSimilarity);
                counter++;
            }

            double score = userVector.mean() + (numerator/denominator);
            scores.set(itemId, score);
        }
    }

    private double CalcSimilarity(SparseVector v1, SparseVector v2)
    {
        // mean centered rating v1
        // a. mean rating of v1
        double meanV1 = v1.mean();
        // b. subtract mean rating from rating vector
        MutableSparseVector mv1 = v1.mutableCopy();
        for(VectorEntry e:mv1.fast()){
            long key = e.getKey();
            double mv1_i = e.getValue();
            mv1_i = mv1_i - meanV1;
            mv1.set(key, mv1_i);
        }

        // mean centered rating v2
        double meanV2 = v2.mean();
        MutableSparseVector mv2 = v2.mutableCopy();
        for(VectorEntry e:mv2.fast()){
            long key = e.getKey();
            double mv2_i = e.getValue();
            mv2_i = mv2_i - meanV2;
            mv2.set(key, mv2_i);
        }

        // now we have the mean-centred rating vectors, calc the cosine of the vectors
        //double numerator = mv1.dot(mv2);
        //double denominator = mv1.norm() * mv2.norm();
        //double cosine = numerator/denominator;

        double cosine = new CosineVectorSimilarity().similarity(mv1, mv2);

        return cosine;
    }


    /**
     * Get a user's rating vector.
     * @param user The user ID.
     * @return The rating vector.
     */
    private SparseVector getUserRatingVector(long user) {
        UserHistory<Rating> history = userDao.getEventsForUser(user, Rating.class);
        if (history == null) {
            history = History.forUser(user);
        }
        return RatingVectorUserHistorySummarizer.makeRatingVector(history);
    }
}


class DataObj
{
    public long userId;
    public double similarity;
}